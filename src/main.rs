mod generator;
mod model;

use burn::{
    backend::{NdArray, Wgpu, LibTorch},
    prelude::*,
};

use clap::{Parser, command};
use generator::InferenceMode;
use std::io;
use std::io::{Read, Write};

use rwkv_tokenizer::WorldTokenizer;

use crate::generator::Generator;
use crate::model::{RWKVv7, RWKVv7Config};

/// Command-line configuration for loading and running an RWKVv7 language model.
///
/// Users can specify model architecture parameters, tokenizer path, weights file,
/// and sampling configuration.
///
/// Example usage:
/// ```sh
/// cargo run -- --n_layer 12 --d_model 768 --n_heads 12 --vocab_size 65536 \
///     --tokenizer_vocab_file ./vocab.txt --weights ./model.safetensors
/// ```
#[derive(Parser, Debug)]
#[command(
    version = "0.1.0",
    author = "dymat",
    about = "Create RWKVv7 model and load weights."
)]
struct Config {
    #[arg(
        short = 'l',
        long = "n_layer",
        default_value_t = 12,
        help = "Number of RWKV layers"
    )]
    n_layer: usize,

    #[arg(
        short = 'd',
        long = "d_model",
        default_value_t = 768,
        help = "Embedding size"
    )]
    d_model: usize,

    #[arg(
        short = 'H',
        long = "n_heads",
        default_value_t = 12,
        help = "Number of attention heads"
    )]
    n_heads: usize,

    #[arg(
        short = 'v',
        long = "vocab_size",
        default_value_t = 65536,
        help = "Vocab size"
    )]
    vocab_size: usize,

    #[arg(
        long = "tokenizer_vocab_file",
        default_value_t = String::from("rwkv_vocab_v20230424.txt"),
        help = "Path to tokenizer vocab file (rwkv_vocab_v20230424.txt)"
    )]
    vocab_path: String,

    #[arg(
        short = 'w',
        long = "weights",
        help = "Path to safetensors weight file"
    )]
    weights: Option<String>,

    #[arg(
        short = 't',
        long = "temperature",
        default_value_t = 0.6,
        help = "Temperature for token sampling"
    )]
    temperature: f32,

    #[arg(
        short = 'p',
        long = "top_p",
        default_value_t = 0.6,
        help = "Top p for token sampling (nucleus)"
    )]
    top_p: f32,

    #[arg(
        short = 'k',
        long = "top_k",
        default_value_t = 50,
        help = "Top k for token sampling"
    )]
    top_k: usize,

    #[arg(
        long = "inference_mode",
        default_value_t = String::from("Mixed"),
        help = "use >Parallel< mode for inference (slower) or >RNN< mode. Default is >Mixed<."
    )]
    inference_mode: String,
}

/// Entry point for the CLI application that loads an RWKVv7 model,
/// initializes a tokenizer, and enables interactive text generation.
///
/// Supports:
/// - Loading weights from safetensors
/// - Resetting state with `\reset`
/// - Exiting with `\exit`
/// - Sampling output using top-k filtering
fn main() {
    let config = Config::parse(); // Parse command-line args

    // Select the backend (NdArray for CPU, can swap for Cuda/Wgpu)
    type MyBackend = LibTorch;
    let device = Default::default();

    // Initialize the model either from weights or config
    let model: RWKVv7<MyBackend>;
    if let Some(weight_path) = config.weights {
        model = RWKVv7::<MyBackend>::new_from_safetensors(&weight_path, &device);
    } else {
        let head_size = config.d_model / config.n_heads;
        model = RWKVv7Config::new(
            config.d_model,
            config.n_heads,
            head_size,
            config.n_layer,
            config.vocab_size,
        )
        .init::<MyBackend>(&device);
    }

    // Load tokenizer from specified vocab file
    let tokenizer =
        WorldTokenizer::new(Some(&config.vocab_path)).expect("Expected to load the tokenizer.");

    // Create text generator using the model and tokenizer
    let mut generator = Generator::new(
        model.clone(), 
        &tokenizer, 
        config.temperature,
        config.top_p,
        config.top_k
    );
    
    match config.inference_mode.to_lowercase().as_str() {
        "mixed" => generator.set_inference_mode(InferenceMode::Mixed),
        "parallel" => generator.set_inference_mode(InferenceMode::Parallel),
        "sequential" => generator.set_inference_mode(InferenceMode::Sequential),
        &_ => {
            println!("Inference Mode '{}' not implemented! Falling back to 'Mixed'", config.inference_mode);
            generator.set_inference_mode(InferenceMode::Mixed);
        },
    };

    // Initial RNN state (will be updated through interaction)
    let mut state = None;

    // Interactive loop
    loop {
        print!("User: ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let trimmed_input = input.trim();

                // Exit command
                if trimmed_input.eq_ignore_ascii_case("\\exit") {
                    break;
                }

                // Reset command
                if trimmed_input.eq_ignore_ascii_case("\\reset") {
                    state = Some(model.get_init_state());
                    println!("Resetting internal model state.");
                    continue;
                }

                // Generate response
                print!("Assistant: ");
                let _ = io::stdout().flush();

                (_, state) = generator.generate(trimmed_input, 64, state);
            }
            Err(e) => println!("Could not read input: {}", e),
        }

        println!("\n");
    }
}
