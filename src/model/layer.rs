use burn::{
    nn::{LayerNorm, LayerNormConfig},
    prelude::*,
    tensor::Tensor,
};

use super::{channel_mix::ChannelMix, time_mix::TimeMix};

/// Stores the internal recurrent state for a transformer layer.
///
/// Used in RNN-style inference to maintain continuity between token steps.
#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    /// Previous input tensor for TimeMix.
    pub tmix_x_prev: Tensor<B, 3>,
    /// Key-value state tensor for TimeMix attention.
    pub tmix_kv: Tensor<B, 3>,
    /// Previous input tensor for ChannelMix.
    pub cmix_x_prev: Tensor<B, 3>,
}

/// A single layer of the RWKVv7 model, composed of TimeMix and ChannelMix blocks
/// along with layer normalizations.
///
/// Each layer operates on its own input and contributes to the recurrent state.
#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    /// The index of this layer in the model stack.
    pub layer_id: usize,

    /// Total number of layers in the model.
    pub n_layer: usize,

    /// Dimensionality of model hidden states.
    pub d_model: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Size of each attention head.
    pub head_size: usize,

    /// LoRA dimension for decay component.
    pub d_decay_lora: usize,

    /// LoRA dimension for AAA component.
    pub d_aaa_lora: usize,

    /// LoRA dimension for MV component.
    pub d_mv_lora: usize,

    /// LoRA dimension for gate component.
    pub d_gate_lora: usize,

    /// First layer normalization before TimeMix.
    pub layer_norm_1: LayerNorm<B>,

    /// TimeMix module for temporal mixing and attention.
    pub tmix: TimeMix<B>,

    /// ChannelMix module for feedforward-style transformation.
    pub cmix: ChannelMix<B>,

    /// Second layer normalization before ChannelMix.
    pub layer_norm_2: LayerNorm<B>,
}

impl<B: Backend> Layer<B> {
    /// Constructs a new layer with initialized TimeMix and ChannelMix modules.
    ///
    /// # Arguments
    ///
    /// * `device` - Target backend device.
    /// * `layer_id` - Index of the layer.
    /// * `n_layer` - Total number of layers in the model.
    /// * `d_model` - Size of the model hidden state.
    /// * `n_heads` - Number of attention heads.
    /// * `head_size` - Size of each head.
    /// * `d_decay_lora`, `d_aaa_lora`, `d_mv_lora`, `d_gate_lora` - LoRA configuration parameters.
    ///
    /// # Returns
    ///
    /// A fully constructed `Layer` instance.
    pub fn new(
        device: &B::Device,
        layer_id: usize,
        n_layer: usize,
        d_model: usize,
        n_heads: usize,
        head_size: usize,
        d_decay_lora: usize,
        d_aaa_lora: usize,
        d_mv_lora: usize,
        d_gate_lora: usize,
    ) -> Layer<B> {
        let tmix = TimeMix::<B>::new(
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
        );
        let cmix = ChannelMix::<B>::new(device, d_model);
        let layer_norm_1 = LayerNormConfig::new(d_model).init::<B>(device);
        let layer_norm_2 = LayerNormConfig::new(d_model).init::<B>(device);
        Layer {
            layer_id,
            n_layer,
            d_model,
            n_heads,
            head_size,
            d_decay_lora,
            d_aaa_lora,
            d_mv_lora,
            d_gate_lora,
            layer_norm_1,
            tmix,
            cmix,
            layer_norm_2,
        }
    }

    /// Executes a parallel forward pass through the layer for a full sequence batch.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch_size, seq_len, d_model]`.
    /// * `v_first` - Optional initial value tensor for attention mixing.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * Output tensor after TimeMix and ChannelMix.
    /// * Optional updated value tensor for attention.
    pub fn forward_parallel(
        &mut self,
        x: Tensor<B, 3>,
        v_first: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 3>>, LayerState<B>) {
        let _x = self.layer_norm_1.forward(x.clone());

        let tmix_prev = _x.clone();        
        let (x_tmix, v_first, tmix_kv) = self.tmix.forward_parallel(_x, v_first);
        
        let _x = x + x_tmix;
        let x_normed = self.layer_norm_2.forward(_x.clone());
        let cmix_prev = x_normed.clone();

        let x = _x.clone() + self.cmix.forward_parallel(x_normed);
        let dims_x = x.dims();

        let layer_state = LayerState::<B>{
            tmix_x_prev: tmix_prev.slice([dims_x[0]-1, dims_x[1]-1]),
            tmix_kv: tmix_kv.reshape([self.n_heads, self.head_size, self.head_size]),
            cmix_x_prev: cmix_prev.slice([dims_x[0]-1, dims_x[1]-1])
        };

        (x, v_first, layer_state)
    }

    /// Executes a forward pass for a single token step in RNN-style inference.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[d_model]`.
    /// * `v_first` - Optional value from previous attention step.
    /// * `tmix_x_prev` - Previous input to TimeMix.
    /// * `tmix_vk_state` - Current key-value state for TimeMix.
    /// * `cmix_x_prev` - Previous input to ChannelMix.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// * Output tensor `[d_model]`.
    /// * Updated value tensor for attention.
    /// * Updated `LayerState` with all internal states.
    pub fn forward_rnn(
        &self,
        x: Tensor<B, 1>,
        v_first: Option<Tensor<B, 1>>,
        tmix_x_prev: Tensor<B, 1>,
        tmix_vk_state: Tensor<B, 3>,
        cmix_x_prev: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Option<Tensor<B, 1>>, LayerState<B>) {
        let _x = self.layer_norm_1.forward(x.clone());
        let (_x, tmix_x_prev, vk_state, v_first) =
            self.tmix
                .forward_rnn(_x, tmix_x_prev, v_first, tmix_vk_state);

        let x = x + _x;
        let _x = self.layer_norm_2.forward(x.clone());

        let (_x, cmix_x_prev) = self.cmix.forward_rnn(_x, cmix_x_prev);
        let x = x + _x;

        (
            x,
            v_first,
            LayerState::<B> {
                tmix_x_prev: tmix_x_prev.reshape([1, 1, self.d_model]),
                tmix_kv: vk_state,
                cmix_x_prev: cmix_x_prev.reshape([1, 1, self.d_model]),
            },
        )
    }
}
