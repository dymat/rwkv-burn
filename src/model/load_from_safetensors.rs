use burn::{
    module::Param,
    nn::GroupNormConfig,
    prelude::*,
    tensor::{Shape, Tensor, TensorData},
};

use safetensors::SafeTensors;
use std::fs;

use crate::model::{RWKVv7, RWKVv7Config};

/// Loads an `RWKVv7` model from a `.safetensors` file.
///
/// This function reads the file from the given path, deserializes it,
/// extracts the model configuration, initializes the model, and populates
/// it with weights from the file.
///
/// # Arguments
/// * `path` - Path to the `.safetensors` file.
/// * `device` - The device (e.g., CPU or GPU) to load the tensors onto.
///
/// # Returns
/// An initialized `RWKVv7` model with all weights loaded.
impl<B: Backend> RWKVv7<B> {
    pub fn new_from_safetensors(path: &str, device: &B::Device) -> RWKVv7<B> {
        let data = fs::read(path).unwrap();
        let safetensors = SafeTensors::deserialize(&data).unwrap();

        let model_config = get_model_config(&safetensors);
        let mut model = model_config.init::<B>(device);

        apply_weights(safetensors, &mut model, device);

        model
    }
}

/// Extracts the model configuration from a `SafeTensors` object.
///
/// Infers key model parameters like vocabulary size, embedding dimension,
/// number of heads, head size, and number of layers by inspecting tensor shapes.
///
/// # Arguments
/// * `safetensors` - Reference to a deserialized `SafeTensors` object.
///
/// # Returns
/// A constructed `RWKVv7Config` with dimensions extracted from the tensors.
fn get_model_config(safetensors: &SafeTensors) -> RWKVv7Config {
    let [vocab_size, d_model] = *safetensors.tensor("emb.weight").unwrap().shape() else {
        todo!()
    };
    let [n_heads, head_size] = *safetensors.tensor("blocks.0.att.r_k").unwrap().shape() else {
        todo!()
    };

    let block_ids = get_block_ids(safetensors);
    let n_layer = *block_ids.iter().max().unwrap() + 1;

    RWKVv7Config::new(d_model, n_heads, head_size, n_layer, vocab_size)
}

/// Retrieves all unique block IDs from the `SafeTensors` keys.
///
/// It looks for tensor names that start with "block" and extracts the block index
/// from the name, ensuring each index appears only once.
///
/// # Arguments
/// * `safetensors` - Reference to a deserialized `SafeTensors` object.
///
/// # Returns
/// A list of unique block indices present in the model file.
fn get_block_ids(safetensors: &SafeTensors) -> Vec<usize> {
    let mut block_ids: Vec<usize> = vec![];
    for (name, _) in safetensors.iter() {
        if name.starts_with("block") {
            let block_id: usize = name.split(".").collect::<Vec<&str>>()[1]
                .parse::<usize>()
                .unwrap();

            if block_ids.iter().find(|item| **item == block_id) == None {
                block_ids.push(block_id);
            }
        }
    }

    block_ids
}

/// Loads a single tensor from a `SafeTensors` object and transfers it to the specified device.
///
/// Automatically converts the tensor from `BF16` to `F32` for computation compatibility.
///
/// # Type Parameters
/// * `B` - The backend to use (e.g., CPU, CUDA).
/// * `D` - The dimensionality of the tensor.
///
/// # Arguments
/// * `safetensors` - Reference to the deserialized `SafeTensors`.
/// * `name` - Name of the tensor to load.
/// * `device` - The target device.
///
/// # Returns
/// A `Tensor<B, D>` loaded and ready on the specified device.
fn get_tensor<B: Backend, const D: usize>(
    safetensors: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, D> {
    let tensor_view = safetensors.tensor(name).unwrap();
    let tensor_data = TensorData::from_bytes(
        tensor_view.data().to_vec(),
        Shape::from(tensor_view.shape().to_vec()),
        burn::tensor::DType::BF16,
    )
    .convert_dtype(burn::tensor::DType::F32);

    Tensor::<B, D>::from_data(tensor_data, device)
}

/// Loads all model weights from the given `SafeTensors` object into the `RWKVv7` model.
///
/// This function populates the embedding, unembedding, layer norms,
/// attention ("time mixing") and feedforward ("channel mixing") weights
/// for each block in the model.
///
/// # Arguments
/// * `safetensors` - The deserialized weights from the model file.
/// * `model` - A mutable reference to the model to populate.
/// * `device` - The device to which all tensors should be loaded.
fn apply_weights<B: Backend>(safetensors: SafeTensors, model: &mut RWKVv7<B>, device: &B::Device) {
    // embedding
    model.embed.weight = Param::from_tensor(get_tensor(&safetensors, "emb.weight", device));

    // unembed
    model.unembed.weight =
        Param::from_tensor(get_tensor(&safetensors, "head.weight", device).transpose());

    // layer norm
    model.layer_norm_in.gamma =
        Param::from_tensor(get_tensor(&safetensors, "blocks.0.ln0.weight", device));
    model.layer_norm_in.beta =
        Param::from_tensor(get_tensor(&safetensors, "blocks.0.ln0.bias", device));

    model.layer_norm_out.gamma =
        Param::from_tensor(get_tensor(&safetensors, "ln_out.weight", device));
    model.layer_norm_out.beta = Param::from_tensor(get_tensor(&safetensors, "ln_out.bias", device));

    /*************************
     * Handle Blocks
     *************************/
    let block_ids: Vec<usize> = get_block_ids(&safetensors);

    for block_id in block_ids.iter() {
        if *block_id >= model.layers.len() {
            println!(
                "There are more layers in weights file ({}/{}) than defined for the model ({})",
                block_id + 1,
                &block_ids.len(),
                model.layers.len()
            );
            continue;
        }

        // layer norm
        model.layers[*block_id].layer_norm_1.gamma = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.ln1.weight", block_id),
            device,
        ));
        model.layers[*block_id].layer_norm_1.beta = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.ln1.bias", block_id),
            device,
        ));

        model.layers[*block_id].layer_norm_2.gamma = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.ln2.weight", block_id),
            device,
        ));
        model.layers[*block_id].layer_norm_2.beta = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.ln2.bias", block_id),
            device,
        ));

        // time mixing
        model.layers[*block_id].tmix.x_r = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_r", block_id),
            device,
        ));
        model.layers[*block_id].tmix.x_w = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_w", block_id),
            device,
        ));
        model.layers[*block_id].tmix.x_k = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_k", block_id),
            device,
        ));
        model.layers[*block_id].tmix.x_v = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_v", block_id),
            device,
        ));
        model.layers[*block_id].tmix.x_a = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_a", block_id),
            device,
        ));
        model.layers[*block_id].tmix.x_g = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.x_g", block_id),
            device,
        ));
        model.layers[*block_id].tmix.w0 = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.w0", block_id),
            device,
        ));
        model.layers[*block_id].tmix.r_k = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.r_k", block_id),
            device,
        ));
        model.layers[*block_id].tmix.a0 = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.a0", block_id),
            device,
        ));
        model.layers[*block_id].tmix.w1.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.w1", block_id),
            device,
        ));
        model.layers[*block_id].tmix.w2.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.w2", block_id),
            device,
        ));
        model.layers[*block_id].tmix.a1.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.a1", block_id),
            device,
        ));
        model.layers[*block_id].tmix.a2.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.a2", block_id),
            device,
        ));
        model.layers[*block_id].tmix.k_k = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.k_k", block_id),
            device,
        ));
        model.layers[*block_id].tmix.g1.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.g1", block_id),
            device,
        ));
        model.layers[*block_id].tmix.g2.weight = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.g2", block_id),
            device,
        ));
        model.layers[*block_id].tmix.k_a = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.k_a", block_id),
            device,
        ));
        model.layers[*block_id].tmix.receptance.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.att.receptance.weight", block_id),
                device,
            )
            .transpose(),
        );
        model.layers[*block_id].tmix.key.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.att.key.weight", block_id),
                device,
            )
            .transpose(),
        );
        model.layers[*block_id].tmix.value.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.att.value.weight", block_id),
                device,
            )
            .transpose(),
        );
        model.layers[*block_id].tmix.output.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.att.output.weight", block_id),
                device,
            )
            .transpose(),
        );

        let mut group_norm = GroupNormConfig::new(model.layers[0].n_heads, model.layers[0].d_model)
            .with_epsilon(64e-5)
            .init::<B>(device);
        group_norm.gamma = Some(Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.ln_x.weight", block_id),
            device,
        )));
        group_norm.beta = Some(Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.att.ln_x.bias", block_id),
            device,
        )));
        model.layers[*block_id].tmix.group_norm = group_norm;

        // channel mixing
        model.layers[*block_id].cmix.key.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.ffn.key.weight", block_id),
                device,
            )
            .transpose(),
        );
        model.layers[*block_id].cmix.value.weight = Param::from_tensor(
            get_tensor(
                &safetensors,
                &format!("blocks.{}.ffn.value.weight", block_id),
                device,
            )
            .transpose(),
        );
        model.layers[*block_id].cmix.x_k = Param::from_tensor(get_tensor(
            &safetensors,
            &format!("blocks.{}.ffn.x_k", block_id),
            device,
        ));

        if *block_id != 0 {
            model.layers[*block_id].tmix.v0 = Param::from_tensor(get_tensor(
                &safetensors,
                &format!("blocks.{}.att.v0", block_id),
                device,
            ));
            model.layers[*block_id].tmix.v1.weight = Param::from_tensor(get_tensor(
                &safetensors,
                &format!("blocks.{}.att.v1", block_id),
                device,
            ));
            model.layers[*block_id].tmix.v2.weight = Param::from_tensor(get_tensor(
                &safetensors,
                &format!("blocks.{}.att.v2", block_id),
                device,
            ));
        }
    }
}
