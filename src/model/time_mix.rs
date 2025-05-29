use burn::{
    module::Param,
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{Tensor, activation, s},
};

/// A module implementing the TimeMix layer for sequential processing.
///
/// This layer combines recurrent-style temporal mixing with attention-like computation,
/// using linear layers and tensor-based modulation. It is optimized for efficient inference
/// in both RNN and parallel modes, often used in RWKV-style architectures.
#[derive(Module, Debug)]
pub struct TimeMix<B: Backend> {
    pub layer_id: usize,
    pub n_layer: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_size: usize,
    pub d_decay_lora: usize,
    pub d_aaa_lora: usize,
    pub d_mv_lora: usize,
    pub d_gate_lora: usize,

    pub x_r: Param<Tensor<B, 3>>,
    pub x_w: Param<Tensor<B, 3>>,
    pub x_k: Param<Tensor<B, 3>>,
    pub x_v: Param<Tensor<B, 3>>,
    pub x_a: Param<Tensor<B, 3>>,
    pub x_g: Param<Tensor<B, 3>>,

    pub w0: Param<Tensor<B, 3>>,
    pub w1: Linear<B>,
    pub w2: Linear<B>,

    pub a0: Param<Tensor<B, 3>>,
    pub a1: Linear<B>,
    pub a2: Linear<B>,

    pub v0: Param<Tensor<B, 3>>,
    pub v1: Linear<B>,
    pub v2: Linear<B>,

    pub g1: Linear<B>,
    pub g2: Linear<B>,

    pub k_k: Param<Tensor<B, 3>>,
    pub k_a: Param<Tensor<B, 3>>,
    pub r_k: Param<Tensor<B, 2>>,

    wkv_op: WKVv7,

    pub receptance: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub group_norm: GroupNorm<B>,
}

impl<B: Backend> TimeMix<B> {
    /// Constructs a new `TimeMix` instance with specified model dimensions and LoRA capacities.
    ///
    /// # Arguments
    /// * `device` - The device on which tensors will be allocated.
    /// * `layer_id` - The index of this layer in the overall model.
    /// * `n_layer` - Total number of layers in the model.
    /// * `d_model` - The dimensionality of the input and output representations.
    /// * `n_heads` - Number of attention heads.
    /// * `head_size` - Dimensionality per attention head.
    /// * `d_decay_lora` - LoRA size for the decay weights.
    /// * `d_aaa_lora` - LoRA size for the attention "a" weights.
    /// * `d_mv_lora` - LoRA size for the value mixing weights.
    /// * `d_gate_lora` - LoRA size for the gating weights.
    ///
    /// # Returns
    /// A fully initialized `TimeMix` module.
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
    ) -> TimeMix<B> {
        let x_r = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));

        let x_w = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let x_k = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let x_v = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let x_a = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let x_g = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));

        let w0 = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let w1 = LinearConfig::new(d_model, d_decay_lora)
            .with_bias(false)
            .init::<B>(device);
        let w2 = LinearConfig::new(d_decay_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let a0 = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let a1 = LinearConfig::new(d_model, d_aaa_lora)
            .with_bias(false)
            .init::<B>(device);
        let a2 = LinearConfig::new(d_aaa_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let v0 = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let v1 = LinearConfig::new(d_model, d_mv_lora)
            .with_bias(false)
            .init::<B>(device);
        let v2 = LinearConfig::new(d_mv_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let g1 = LinearConfig::new(d_model, d_gate_lora)
            .with_bias(false)
            .init::<B>(device);
        let g2 = LinearConfig::new(d_gate_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let k_k = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let k_a = Param::from_tensor(Tensor::<B, 3>::empty([1, 1, d_model], device));
        let r_k = Param::from_tensor(Tensor::<B, 2>::empty([n_heads, head_size], device));

        //time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        let receptance = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let key = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let value = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let output = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let group_norm = GroupNormConfig::new(n_heads, d_model)
            .with_epsilon(64e-5)
            .init(device);

        let wkv_op = WKVv7::new(n_heads, head_size);

        TimeMix {
            layer_id,
            n_layer,
            d_model,
            n_heads,
            head_size,
            d_decay_lora,
            d_aaa_lora,
            d_mv_lora,
            d_gate_lora,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            v0,
            v1,
            v2,
            g1,
            g2,
            k_k,
            k_a,
            r_k,
            wkv_op,
            receptance,
            key,
            value,
            output,
            group_norm,
        }
    }

    /// Performs a forward pass over a batch of sequences in parallel mode.
    ///
    /// Applies temporal mixing and attention computation across time steps.
    /// This method is optimized for training or evaluation on batched sequence data.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch_size, time_steps, d_model]`.
    /// * `v_first` - Optional initial value vector for value mixing across steps.
    ///
    /// # Returns
    /// A tuple containing:
    /// * The output tensor of shape `[batch_size, time_steps, d_model]`.
    /// * The updated or original `v_first` tensor.
    pub fn forward_parallel(
        &self,
        x: Tensor<B, 3>,
        mut v_first: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 3>>, Tensor<B, 4>) {
        let [batch_size, time_steps, _] = x.clone().dims();

        let xx = x
            .clone()
            .slice([0..batch_size, 0..time_steps - 1])
            .pad((0, 0, 1, 0), 0);

        let xx = xx - x.clone();

        let xr = x.clone() + xx.clone().mul(self.x_r.val());
        let xw = x.clone() + xx.clone().mul(self.x_w.val());
        let xk = x.clone() + xx.clone().mul(self.x_k.val());
        let xv = x.clone() + xx.clone().mul(self.x_v.val());
        let xa = x.clone() + xx.clone().mul(self.x_a.val());
        let xg = x.clone() + xx.clone().mul(self.x_g.val());

        let r = self.receptance.forward(xr);
        let w = -activation::softplus(
            -(self.w0.val() + self.w2.forward(activation::tanh(self.w1.forward(xw)))),
            1.0,
        ) - 0.5;

        let mut v = self.value.forward(xv.clone());

        if let Some(_v_first) = v_first.clone() {
            v = v.clone()
                + (_v_first.clone() - v.clone()).mul(activation::sigmoid(
                    self.v0.val() + self.v2.forward(self.v1.forward(xv.clone())),
                ));
        } else {
            v_first = Some(v.clone());
        }

        let a = activation::sigmoid(self.a0.val() + self.a2.forward(self.a1.forward(xa)));
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        let k = self.key.forward(xk);
        let kk = k.clone().mul(self.k_k.val());
        let kk = kk.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let kk = (kk.clone().div(kk.clone().mul(kk.clone()).sum_dim(3))).reshape([
            batch_size,
            time_steps,
            self.d_model,
        ]);
        let k = k
            .clone()
            .mul((a.clone() - 1).mul(self.k_a.val()).add_scalar(1.0));

        let (x, state)=
            self.wkv_op
                .clone()
                .forward(r.clone(), w, k.clone(), v.clone(), -kk.clone(), kk.mul(a));

        let x = self
            .group_norm
            .forward(x.reshape([batch_size * time_steps, self.d_model]))
            .reshape([batch_size, time_steps, self.d_model]);

        let x = x
            + ((r
                .reshape([batch_size, time_steps, self.n_heads, self.head_size])
                .mul(
                    k.reshape([batch_size, time_steps, self.n_heads, self.head_size])
                        .mul(self.r_k.val().unsqueeze_dims(&[0, 1])),
                ))
            .sum_dim(3)
            .mul(v.reshape([
                batch_size,
                time_steps,
                self.n_heads,
                self.head_size,
            ])))
            .reshape([batch_size, time_steps, self.d_model]);
        let x = self.output.forward(x.mul(g));

        (x, v_first, state)
    }

    /// Performs a forward pass over a single time step in recurrent (RNN) mode.
    ///
    /// This is used during inference when sequence elements are processed one at a time.
    /// Maintains internal state across time steps for temporal consistency.
    ///
    /// # Arguments
    /// * `x` - Current input vector at time `t`, shape `[d_model]`.
    /// * `x_prev` - Previous input vector at time `t-1`, shape `[d_model]`.
    /// * `v_first` - Optional reference value for value mixing.
    /// * `vk_state` - Accumulated attention state, shape `[n_heads, head_size, head_size]`.
    ///
    /// # Returns
    /// A tuple containing:
    /// * The new output vector, shape `[d_model]`.
    /// * The current input.
    /// * The updated attention state.
    /// * The updated or original `v_first` value.
    pub fn forward_rnn(
        &self,
        x: Tensor<B, 1>,
        x_prev: Tensor<B, 1>,
        mut v_first: Option<Tensor<B, 1>>,
        vk_state: Tensor<B, 3>,
    ) -> (
        Tensor<B, 1>,         // x_new
        Tensor<B, 1>,         // x
        Tensor<B, 3>,         // vk_state
        Option<Tensor<B, 1>>, // v_first
    ) {
        let x = x.reshape([self.d_model]);
        let x_prev = x_prev.reshape([self.d_model]);

        let xx = x_prev - x.clone();

        let xr = x.clone() + xx.clone().mul(self.x_r.val().reshape([self.d_model]));
        let xw = x.clone() + xx.clone().mul(self.x_w.val().reshape([self.d_model]));
        let xk = x.clone() + xx.clone().mul(self.x_k.val().reshape([self.d_model]));
        let xv = x.clone() + xx.clone().mul(self.x_v.val().reshape([self.d_model]));
        let xa = x.clone() + xx.clone().mul(self.x_a.val().reshape([self.d_model]));
        let xg = x.clone() + xx.clone().mul(self.x_g.val().reshape([self.d_model]));

        let r = self.receptance.forward(xr);
        let w = self.w2.forward(activation::tanh(self.w1.forward(xw)));
        let k = self.key.forward(xk);
        let mut v = self.value.forward(xv.clone());
        let a = activation::sigmoid(
            self.a0.val().reshape([self.d_model]) + self.a2.forward(self.a1.forward(xa)),
        );
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        let kk = k.clone().mul(self.k_k.val().reshape([self.d_model]));
        let kk = kk.reshape([self.n_heads, self.head_size]);
        let kk = (kk.clone().div(kk.clone().mul(kk.clone()).sum_dim(1))).reshape([self.d_model]);
        let k = k.clone().mul(
            (a.clone() - 1)
                .mul(self.k_a.val().reshape([self.d_model]))
                .add_scalar(1.0),
        );

        if let Some(_v_first) = v_first.clone() {
            v = v.clone()
                + (_v_first - v.clone()).mul(activation::sigmoid(
                    self.v0.val().reshape([self.d_model])
                        + self.v2.forward(self.v1.forward(xv.clone())),
                ));
        } else {
            v_first = Some(v.clone());
        }

        let w = w + self.w0.val().reshape([self.d_model]);
        let w = activation::sigmoid(w).mul_scalar(-0.606531).exp();

        let vk = v
            .clone()
            .reshape([self.n_heads, self.head_size, 1])
            .matmul(k.clone().reshape([self.n_heads, 1, self.head_size]));

        let ab = (-kk.clone())
            .reshape([self.n_heads, self.head_size, 1])
            .matmul(kk.clone().mul(a).reshape([self.n_heads, 1, self.head_size]));

        let vk_state = vk_state
            .clone()
            .mul(w.reshape([self.n_heads, 1, self.head_size]))
            + vk_state.clone().matmul(ab)
            + vk;

        let out = vk_state
            .clone()
            .matmul(r.clone().reshape([self.n_heads, self.head_size, 1]));

        let out = self
            .group_norm
            .forward(out.reshape([1, self.d_model]))
            .reshape([self.d_model]);

        let out = out
            + r.mul(k)
                .mul(self.r_k.val().reshape([self.d_model]))
                .reshape([self.n_heads, self.head_size])
                .sum_dim(1)
                .mul(v.reshape([self.n_heads, self.head_size]))
                .reshape([self.d_model]);

        let out = self.output.forward(out.mul(g));

        (out, x, vk_state, v_first)
    }
}

/// A helper struct implementing the custom WKV attention operator.
///
/// Performs a time-recursive computation used for efficient memory-based attention,
/// replacing standard softmax attention in RWKV models.
#[derive(Module, Clone, Debug)]
struct WKVv7 {
    n_heads: usize,
    head_size: usize,
}

impl WKVv7 {
    /// Creates a new instance of the WKVv7 operator.
    ///
    /// # Arguments
    /// * `n_heads` - Number of attention heads.
    /// * `head_size` - Size per attention head.
    ///
    /// # Returns
    /// A new `WKVv7` struct.
    fn new(n_heads: usize, head_size: usize) -> WKVv7 {
        WKVv7 { n_heads, head_size }
    }

    /// Computes the weighted key-value aggregation over time using recursive computation.
    ///
    /// This function operates on reshaped 4D tensors and computes a form of time-aware
    /// attention based on decaying state accumulation.
    ///
    /// # Arguments
    /// * `r` - Receptance tensor of shape `[B, T, d_model]`.
    /// * `w` - Decay weights tensor.
    /// * `k` - Key tensor.
    /// * `v` - Value tensor.
    /// * `a` - Pre-activation attention modifier tensor.
    /// * `b` - Attention booster tensor.
    ///
    /// # Returns
    /// A tuple containing:
    /// * Aggregated output tensor of shape `[B, T, d_model]`.
    /// * State tensor of shape `[B, T, head_size, head_size]`, that can be used for further inference in RNN mode.
    fn forward<B: Backend>(
        self,
        r: Tensor<B, 3>,
        w: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        a: Tensor<B, 3>,
        b: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let [batch_size, time_steps, d_model] = r.clone().dims();

        let r: Tensor<B, 4> = r.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let k: Tensor<B, 4> = k.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let v: Tensor<B, 4> = v.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let a: Tensor<B, 4> =
            a.clone()
                .reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let b: Tensor<B, 4> = b.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let w: Tensor<B, 4> = Tensor::exp(-Tensor::exp(w.reshape([
            batch_size,
            time_steps,
            self.n_heads,
            self.head_size,
        ])));

        let mut out: Tensor<B, 4> = Tensor::zeros(
            [batch_size, time_steps, self.n_heads, self.head_size],
            &r.device(),
        );
        let mut state: Tensor<B, 4> = Tensor::zeros(
            [batch_size, self.n_heads, self.head_size, self.head_size],
            &r.device(),
        );

        for t in 0..time_steps {
            let kk =
                k.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, 1, self.head_size]);
            let rr =
                r.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let vv =
                v.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let aa =
                a.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let bb =
                b.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, 1, self.head_size]);
            let ww =
                w.clone()
                    .slice(s![.., t])
                    .reshape([batch_size, self.n_heads, 1, self.head_size]);

            state = state.clone().mul(ww) + state.clone().matmul(aa.matmul(bb)) + vv.matmul(kk);

            out = out.slice_assign(
                [0..batch_size, t..t + 1, 0..self.n_heads, 0..self.head_size],
                state
                    .clone()
                    .matmul(rr)
                    .reshape([batch_size, 1, self.n_heads, self.head_size]),
            );
        }
        (out.reshape([batch_size, time_steps, d_model]), state)
    }
}
