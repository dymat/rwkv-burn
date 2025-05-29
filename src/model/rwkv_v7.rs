use burn::{
    config::Config,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::Tensor,
};

use super::layer::{Layer, LayerState};

/// Configuration struct for the RWKVv7 model.
///
/// Includes model size, number of layers and heads, vocabulary size,
/// and optional dimensions for LoRA-based modules.
#[derive(Config, Debug)]
pub struct RWKVv7Config {
    /// Size of the embedding vector and model dimension.
    pub d_model: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Dimensionality of each attention head.
    pub head_size: usize,

    /// Number of transformer layers.
    pub n_layer: usize,

    /// Size of the token vocabulary.
    pub vocab_size: usize,

    /// Dimension for LoRA decay component (default: 64).
    #[config(default = 64)]
    pub d_decay_lora: usize,

    /// Dimension for LoRA AAA component (default: 64).
    #[config(default = 64)]
    pub d_aaa_lora: usize,

    /// Dimension for LoRA MV component (default: 32).
    #[config(default = 32)]
    pub d_mv_lora: usize,

    /// Dimension for LoRA gate component (default: 128).
    #[config(default = 128)]
    pub d_gate_lora: usize,
}

impl RWKVv7Config {
    /// Initializes an `RWKVv7` model from the configuration on the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - Target backend device for tensor allocation (e.g., CPU or GPU).
    ///
    /// # Returns
    ///
    /// A fully constructed `RWKVv7` model instance.
    pub fn init<B: Backend>(self, device: &B::Device) -> RWKVv7<B> {
        RWKVv7::new(
            device,
            self.vocab_size,
            self.d_model,
            self.n_heads,
            self.head_size,
            self.n_layer,
            self.d_decay_lora,
            self.d_aaa_lora,
            self.d_mv_lora,
            self.d_gate_lora,
        )
    }
}

/// The RWKVv7 neural network model.
///
/// This model uses embedding layers, layer normalization, a stack of
/// custom `Layer`s, and a final projection layer to map hidden states
/// back to vocabulary logits.
#[derive(Module, Debug)]
pub struct RWKVv7<B: Backend> {
    /// Dimensionality of the model and hidden layers.
    d_model: usize,

    /// Number of attention heads in each transformer layer.
    n_heads: usize,

    /// Size of each individual attention head.
    head_size: usize,

    /// Token embedding layer.
    pub embed: Embedding<B>,
    /// Input layer normalization.
    pub layer_norm_in: LayerNorm<B>,

    /// Vector of transformer layers.
    pub layers: Vec<Layer<B>>,

    /// Final output layer normalization.
    pub layer_norm_out: LayerNorm<B>,
    /// Final linear projection to vocabulary size.
    pub unembed: Linear<B>,
}

impl<B: Backend> RWKVv7<B> {
    /// Creates a new RWKVv7 model with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The backend device (e.g., CPU or GPU) for tensor operations.
    /// * `vocab_size` - Size of the vocabulary for embedding and output projection.
    /// * `d_model` - Dimensionality of the model (embedding size).
    /// * `n_heads` - Number of attention heads.
    /// * `head_size` - Size of each attention head.
    /// * `n_layer` - Number of transformer layers.
    /// * `d_decay_lora`, `d_aaa_lora`, `d_mv_lora`, `d_gate_lora` - LoRA dimensions for different components.
    ///
    /// # Returns
    ///
    /// A new instance of `RWKVv7`.
    fn new(
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        head_size: usize,
        n_layer: usize,
        d_decay_lora: usize,
        d_aaa_lora: usize,
        d_mv_lora: usize,
        d_gate_lora: usize,
    ) -> RWKVv7<B> {
        let embed = EmbeddingConfig::new(vocab_size, d_model).init::<B>(device);
        let layer_norm_in = LayerNormConfig::new(d_model).init::<B>(device);
        let layer_norm_out = LayerNormConfig::new(d_model).init::<B>(device);
        let unembed = LinearConfig::new(d_model, vocab_size)
            .with_bias(false)
            .init::<B>(device);

        let layers: Vec<Layer<B>> = (0..n_layer)
            .map(|layer_id| {
                Layer::new(
                    device,
                    layer_id,
                    n_layer,
                    d_model,
                    n_heads,
                    head_size,
                    d_decay_lora,
                    d_aaa_lora,
                    d_mv_lora,
                    d_gate_lora,
                )
            })
            .collect();

        RWKVv7 {
            d_model,
            n_heads,
            head_size,
            embed,
            layer_norm_in,
            layers,
            layer_norm_out,
            unembed,
        }
    }

    /// Runs a parallel forward pass on a batch of input token sequences.
    ///
    /// # Arguments
    ///
    /// * `x` - A 2D tensor of integer token IDs with shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// A 3D tensor with shape `[batch_size, seq_len, vocab_size]` containing the model logits.
    pub fn forward_parallel(&mut self, x: Tensor<B, 2, burn::tensor::Int>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        let x = self.embed.forward(x);

        let mut x = self.layer_norm_in.forward(x);
        let mut v_first: Option<Tensor<B, 3>> = None;

        let mut layer_states = self.get_init_state();

        for i in 0..self.layers.len() {
            (x, v_first, layer_states[i]) = self.layers[i].forward_parallel(x, v_first);
        }

        let x = self.layer_norm_out.forward(x);

        // return the logits for the last token (which is the prediction of the new token) 
        (
            self.unembed.forward(x.slice(s![.., -1])),
            layer_states
        )
    }

    /// Runs a forward pass on a single token in RNN-style inference.
    ///
    /// # Arguments
    ///
    /// * `x` - A single token ID as `i32`.
    /// * `state` - Current hidden states for each transformer layer.
    ///
    /// # Returns
    ///
    /// A tuple containing the output logits and updated layer states.
    pub fn forward_rnn(
        &mut self,
        token: i32,
        mut state: Vec<LayerState<B>>,
    ) -> (Tensor<B, 1>, Vec<LayerState<B>>) {
        let x = self
            .embed
            .weight
            .val()
            .clone()
            .slice(token)
            .reshape([self.d_model]);
        let mut x = self.layer_norm_in.forward(x);
        let mut v_first: Option<Tensor<B, 1>> = None;

        for i in 0..self.layers.len() {
            let layer_state: LayerState<B>;
            (x, v_first, layer_state) = self.layers[i].forward_rnn(
                x,
                v_first,
                state[i].tmix_x_prev.clone().reshape([self.d_model]),
                state[i].tmix_kv.clone(),
                state[i].cmix_x_prev.clone().reshape([self.d_model]),
            );

            state[i] = layer_state;
        }

        (self.unembed.forward(self.layer_norm_out.forward(x)), state)
    }

    /// Returns the initial hidden state for each transformer layer.
    ///
    /// # Returns
    ///
    /// A vector of `LayerState` with zero-initialized tensors for temporal and channel mixing.
    pub fn get_init_state(&self) -> Vec<LayerState<B>> {
        (0..self.layers.len())
            .map(|_| LayerState {
                tmix_x_prev: Tensor::<B, 3>::zeros(
                    [1, 1, self.d_model],
                    &self.embed.weight.device(),
                ),
                tmix_kv: Tensor::<B, 3>::zeros(
                    [self.n_heads, self.head_size, self.head_size],
                    &self.embed.weight.device(),
                ),
                cmix_x_prev: Tensor::<B, 3>::zeros(
                    [1, 1, self.d_model],
                    &self.embed.weight.device(),
                ),
            })
            .collect::<Vec<LayerState<B>>>()
    }
}
