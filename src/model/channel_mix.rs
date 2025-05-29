use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{Tensor, activation},
};

/// A feedforward (channel mixing) layer used in RWKV models.
///
/// This struct performs a per-token nonlinear transformation using a two-layer
/// MLP with ReLU squared activation, optionally conditioned on previous input
/// for RNN-style recurrence.
///
/// # Type Parameters
/// * `B` - The backend to use (e.g., NdArray or LibTorch).
#[derive(Module, Debug)]
pub struct ChannelMix<B: Backend> {
    /// Model dimensionality (hidden size).
    pub d_model: usize,
    /// Hidden size of the feedforward layer (usually 4 * d_model).
    pub dim_ffn: usize,
    /// Learnable tensor used to scale the delta between current and previous input.
    pub x_k: Param<Tensor<B, 3>>,
    /// Linear layer for projecting to the intermediate (hidden) dimension.
    pub key: Linear<B>,
    /// Linear layer for projecting back to the original model dimension.
    pub value: Linear<B>,
}

impl<B: Backend> ChannelMix<B> {
    /// Constructs a new `ChannelMix` module with initialized parameters.
    ///
    /// # Arguments
    /// * `device` - The target device to allocate tensors on.
    /// * `d_model` - The model's hidden dimension size.
    ///
    /// # Returns
    /// A new instance of `ChannelMix`.
    pub fn new(device: &B::Device, d_model: usize) -> ChannelMix<B> {
        let dim_ffn = 4 * d_model;
        let x_k = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let key = LinearConfig::new(d_model, dim_ffn)
            .with_bias(false)
            .init::<B>(device);
        let value = LinearConfig::new(dim_ffn, d_model)
            .with_bias(false)
            .init::<B>(device);

        ChannelMix {
            d_model,
            dim_ffn,
            x_k,
            key,
            value,
        }
    }

    /// Forward pass of the channel mixing layer in parallel (non-recurrent) mode.
    ///
    /// Used during training or inference with full sequences. The input is mixed
    /// with its temporally shifted version using a learnable gating mechanism.
    ///
    /// # Arguments
    /// * `x` - A 3D tensor of shape `[batch_size, time_steps, d_model]`.
    ///
    /// # Returns
    /// A 3D tensor of shape `[batch_size, time_steps, d_model]` representing the transformed output.
    ///
    pub fn forward_parallel(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, time_steps, _] = x.clone().dims();

        let xx = x
            .clone()
            .slice([0..batch_size, 0..time_steps - 1])
            .pad((0, 0, 1, 0), 0);
        let xx = xx - x.clone();

        let k = activation::relu(self.key.forward(x.clone() + xx.mul(self.x_k.val())));
        let k = k.clone().mul(k);

        self.value.forward(k)
    }

    /// Forward pass in recurrent mode, used for step-by-step inference.
    ///
    /// Processes one token at a time by comparing the current and previous hidden states.
    ///
    /// # Arguments
    /// * `x` - The current input tensor of shape `[d_model]`.
    /// * `x_prev` - The previous input tensor of shape `[d_model]`.
    ///
    /// # Returns
    /// A tuple `(output, updated_state)` where:
    /// - `output` is the transformed tensor for this time step.
    /// - `updated_state` is the new value to be used as `x_prev` in the next step.
    pub fn forward_rnn(
        &self,
        x: Tensor<B, 1>,
        x_prev: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let xx = x_prev - x.clone();
        let k = x.clone() + xx.mul(self.x_k.val().reshape([self.d_model]));
        let k = activation::relu(self.key.forward(k)).powf_scalar(2.0);

        (self.value.forward(k), x)
    }
}
