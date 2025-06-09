use std::{
    io::{self, Write},
    str::Utf8Error,
};

use burn::{
    prelude::*,
    tensor::{DType, Tensor, activation},
};
use rand::prelude::*;
use rwkv_tokenizer::WorldTokenizer;

use crate::model::{LayerState, RWKVv7};

/// Defines the inference mode for text generation.
#[derive(Debug, Clone, Copy)]
pub enum InferenceMode {
    Parallel, 
    Sequential,
    Mixed // Default
}

/// A text generator based on the RWKVv7 model, using a tokenizer and
/// supporting both sequential and parallel inference.
///
/// # Type Parameters
/// * `B` - The backend to use for tensor computation (e.g., NdArray).
#[derive(Debug, Clone)]
pub struct Generator<'a, B: Backend> {
    model: RWKVv7<B>,
    inference_mode: InferenceMode,
    tokenizer: &'a WorldTokenizer,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rng: rand::rngs::ThreadRng,
}

impl<'a, B: Backend> Generator<'a, B> {
    /// Creates a new `Generator` with default sampling settings.
    ///
    /// # Arguments
    /// * `model` - An instance of the RWKVv7 language model.
    /// * `tokenizer` - A reference to a tokenizer used for encoding/decoding text.
    /// * `top_k` - Optional value for top-k sampling.
    ///
    /// # Returns
    /// A `Generator` instance ready for text generation.
    pub fn new(model: RWKVv7<B>, tokenizer: &'a WorldTokenizer, temperature: f32, top_p: f32,  top_k: usize) -> Self {
        Self {
            model,
            inference_mode: InferenceMode::Mixed,
            tokenizer,
            top_k,
            top_p,
            temperature,
            rng: rand::rng(),
        }
    }

    /// Sets the inference mode (`Parallel` or `Sequential`) for text generation.
    ///
    /// # Arguments
    /// * `inference_mode` - Desired inference mode.
    pub fn set_inference_mode(&mut self, inference_mode: InferenceMode) {
        self.inference_mode = inference_mode;
    }

    /// Returns the current inference mode.
    ///
    /// # Returns
    /// The current `InferenceMode` being used.
    pub fn get_inference_mode(&self) -> InferenceMode {
        self.inference_mode
    }

    /// Generates text based on the given prompt and maximum token count.
    ///
    /// Chooses the forward path (parallel or sequential) depending on the selected
    /// inference mode. Uses provided state or initializes from scratch.
    ///
    /// # Arguments
    /// * `prompt` - Initial input string to condition the generation.
    /// * `max_new_tokens` - Number of tokens to generate.
    /// * `state` - Optional RNN state to continue from previous generation.
    ///
    /// # Returns
    /// A tuple of `(generated_string, final_state)`.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        state: Option<Vec<LayerState<B>>>,
    ) -> (String, Option<Vec<LayerState<B>>>) {

        // apply prompt template
        let prompt = format!("User: {}\n\nAssistant:", prompt.trim());
        
        // inference mode selection
        match self.inference_mode {
            // transformer like inference (slow)
            InferenceMode::Parallel => {
                let (completion, state) = self.generate_parallel(&prompt, max_new_tokens);

                (completion, Some(state))
            },
            // RNN like inference (fast)
            InferenceMode::Sequential => {
                let mut _state: Vec<LayerState<B>>;

                if let Some(state) = state {
                    _state = state;
                } else {
                    _state = self.model.get_init_state();
                }

                let (completion, _state) = self.generate_sequential(&prompt, max_new_tokens, _state);

                (completion, Some(_state))
            },
            // mixed inference (default)
            // first parallel, then sequential
            InferenceMode::Mixed => {
                if let Some(state) = state {
                    let (completion, state) = self.generate_sequential(&prompt, max_new_tokens, state);

                    (completion, Some(state))

                } else {
                    let (completion, state) = self.generate_parallel(&prompt, 1);
                    let (completion, state) = self.generate_sequential(&completion, max_new_tokens-1, state);

                    (completion, Some(state))
                }
            }
        }
    }

    /// Converts a prompt string into a tensor suitable for model input.
    ///
    /// # Arguments
    /// * `prompt` - The input text to encode.
    ///
    /// # Returns
    /// A 2D tensor of token IDs with shape `[1, seq_len]`.
    fn prompt_to_tensor(&self, prompt: &str) -> Tensor<B, 2, Int> {
        let mut tokens: Vec<i32> = self
            .tokenizer
            .encode(prompt)
            .iter()
            .map(|&item| item as i32)
            .collect();

        tokens.insert(0, 0); // pad with zero token; => better inference results

        Tensor::<B, 1, Int>::from_data(&tokens[..], &self.model.embed.weight.device())
            .reshape([1, -1])
    }

    /// Encodes a prompt string into a vector of token IDs.
    ///
    /// # Arguments
    /// * `prompt` - The input text to encode.
    ///
    /// # Returns
    /// A vector of token IDs.
    fn prompt_to_vec(&self, prompt: &str) -> Vec<i32> {
        self.tokenizer
            .encode(prompt)
            .iter()
            .map(|&item| item as i32)
            .collect()
    }

    /// Generates text sequentially (token by token), suitable for RNNs.
    ///
    /// # Arguments
    /// * `prompt` - The initial input text.
    /// * `max_new_tokens` - Maximum number of tokens to generate.
    /// * `state` - Initial hidden state.
    ///
    /// # Returns
    /// A tuple `(generated_text, updated_state)`.
    fn generate_sequential(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        state: Vec<LayerState<B>>,
    ) -> (String, Vec<LayerState<B>>) {
        let x = self.prompt_to_vec(prompt);

        let mut state = state;
        let mut y: Tensor<B, 1>;

        let mut last_token: i32 = 0;

        let mut out = "".to_string();

        for token in x {
            (y, state) = self.model.forward_rnn(token, state);

            if let (Ok(string), token, _) = self.sample_next_token(y) {
                last_token = token as i32;
                out = string;
            }
        }

        print!("{}", &out);

        for _ in 0..max_new_tokens {
            (y, state) = self.model.forward_rnn(last_token, state);

            if let (Ok(string), token, _) = self.sample_next_token(y) {
                last_token = token as i32;
                out += &string;

                print!("{}", &string);
                let _ = io::stdout().flush();

                // stop at EndOfText token
                if token == 0 {
                    break;
                }
            }
        }

        (out, state)
    }

    /// Placeholder for future implementation of parallel (non-recurrent) inference.
    ///
    /// # Arguments
    /// * `prompt` - Input prompt text.
    /// * `max_new_tokens` - Number of tokens to generate.
    /// * `state` - Initial hidden state.
    ///
    /// # Returns
    /// Placeholder string and empty state.
    fn generate_parallel(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> (String, Vec<LayerState<B>>) {

        let mut out = prompt.to_string();
        let mut state: Vec<LayerState<B>> = Vec::<LayerState<B>>::with_capacity(self.model.layers.len());
        let mut y = Tensor::<B, 3>::empty([1,1,1], &self.model.embed.weight.device());

        for _ in 0..max_new_tokens {
            let x = self.prompt_to_tensor(&out);
            (y, state) = self.model.forward_parallel(x);
            if let (Ok(string), token, _) = self.sample_next_token(y.reshape([-1])) {
                out += &string;

                print!("{}", &string);
                let _ = io::stdout().flush();

                if token == 0 {
                    break;
                }
            }
        }

        (out, state)
    }

    /// Samples the next token from a model output distribution using temperature, top-k and top-p filtering
    /// and multinomial sampling.
    ///
    /// # Arguments
    /// * `tensor` - A 1D tensor representing the logits.
    ///
    /// # Returns
    /// A tuple containing:
    /// * A sampled token.
    /// * The corresponding probability.
    fn sample_next_token(&mut self, tensor: Tensor<B, 1>) -> (Result<String, Utf8Error>, u16, f32) {
        let mut result: Tensor<B, 1> = tensor;
        let indices: Tensor<B, 1, Int>;

        // apply temperature
        result = result.div_scalar(self.temperature);

        // softmax
        result = activation::softmax(result, 0);

        // top k
        (result, indices) = result.topk_with_indices(self.top_k, 0);
        let mut tokens = indices
            .to_data()
            .convert_dtype(DType::U16)
            .to_vec::<u16>()
            .unwrap();

        let mut probs = result
            .to_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .unwrap();

        // top p
        if self.top_p < 1.0 {
            let mut acc: f32 = 0.0;
            let boundary: usize = probs
                .iter()
                .map(|p| {
                    acc += p;
                    acc
                })
                .take_while(|&cumsum| cumsum <= self.top_p)
                .count();
            
            if boundary > 0 {
                tokens = tokens[0..boundary].to_vec();
                probs = probs[0..boundary].to_vec();
            } else {
                tokens = tokens[0..1].to_vec();
                probs = probs[0..1].to_vec();
            }
        }

        // multinomial sampling
        let items: Vec<(u16, f32)> = tokens.into_iter()
            .zip(probs.into_iter())
            .collect();

        let choice = items
            .choose_weighted(&mut self.rng, |item| item.1);

        if let Ok((token, prob)) = choice {
            (
                self.tokenizer.decode(vec![*token]), 
                *token, 
                *prob
            )
        } else {
            // Fallback in case of sampling failure
            let token = items[0].0;
            let prob = items[0].1;
            
            (
                self.tokenizer.decode(vec![token]),
                token,
                prob,
            )
        }
    }
}
