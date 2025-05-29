# RWKVv7 CLI Inference (Burn Framework)

> A port of the [RWKV v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7) language model, implemented in [Rust](https://www.rust-lang.org/) with the [Burn](https://burn.dev) deep learning framework.

![rwkv](https://img.shields.io/badge/RWKV-v7-blue)
![burn](https://img.shields.io/badge/Burn-ML%20Framework-orange)
![rust](https://img.shields.io/badge/Rust-stable-informational)

---

## ✨ Features

- 🔥 **Pure Burn**: built with [Burn](https://burn.dev)
- 🔤 **Tokenizer** support via [`rwkv-tokenizer`](https://crates.io/crates/rwkv-tokenizer)
- 🧠 Supports **stateful sequential** or **parallel generation** or **mixed generation**
- 🧪 Sampling with **top-k**, **temperature**, and **top-p**
- ⚙️ Load RWKV v7 models from [`safetensors`](https://github.com/huggingface/safetensors)

---

## Help wanted

Currently best results are achieved with [RWKV v7 World 0.1B Model](https://huggingface.co/BlinkDL/rwkv-7-world). 

I appriciate any help investigating the observed performance drop with larger models (0.4B, 1.5B, ...).


## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/dymat/rwkv-burn.git
cd rwkv-burn
cargo build --release
```

### Download and convert model weigths

Download the model weights from Huggingface: https://huggingface.co/BlinkDL/rwkv-7-world

Convert them into SafeTensors format:

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install safetensors
python weights_to_safetensors.py path/to/model_weights.pth
```


## 🧠 Usage

### Download weights and convert to safetensors

```bash
wget https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
python weights_to_safetensors.py RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
```

### Run the CLI

```bash
cargo run --release -- \
  --weights /path/to/model_weights.pth.safetensors \
  --top_p 0.6
  --temperature 0.8
```

## ⚙️ CLI Options

| Flag                        | Description                           | Default     |
|-----------------------------|---------------------------------------|-------------|
| `-l`, `--n_layer`           | Number of RWKV layers                 | `12`        |
| `-d`, `--d_model`           | Embedding (hidden) size               | `768`       |
| `-H`, `--n_heads`           | Number of attention heads             | `12`        |
| `-v`, `--vocab_size`        | Vocabulary size                       | `65536`     |
| `-w`, `--weights`           | Path to `.safetensors` weight file    | _optional_  |
| `-t`, `--temperature`       | Temperature for token sampling        | `0.6`       |
| `-p`, `--top_p`             | Top-p sampling parameter              | `0.6`       |
| `-k`, `--top_k`             | Top-k sampling parameter              | `50`        |
| ``, `--inference_mode`      | Parallel, RNN, Mixed                  | `Mixed`     |
| ``, `--tokenizer_vocab_file`| Path to tokenizer vocab file          | `rwkv_vocab_v20230424.txt` |


## 🛠️ Developer Notes

This project implements a minimal RWKV-v7 model inference pipeline using the [Burn](https://burn.dev) deep learning framework and [rwkv-tokenizer](github.com/cahya-wirawan/rwkv-tokenizer) for tokenization.

### Structure

- `main.rs`: Entry point with CLI parsing and REPL loop.
- `model`: Implements the RWKVv7 model architecture using `burn` modules.
- `generator.rs`: Handles text generation (sequential and parallel modes).
- `rwkv_vocab_v20230424.txt`: Vocabulary file used by the tokenizer.
- `weights_to_safetensors.py`: Optional script to convert pre-trained weights into `.safetensors` format.

### Supported Features

- 🔄 RNN-style token-by-token generation
- 🔥 Top-k sampling (Top-p planned)
- 🧠 Inference state caching across generations
- 🎛 CLI configuration for model size, heads, tokenizer, weights


### 🛤️ Roadmap

This project is under active development. Below are the planned features and improvements:

#### Short-term Goals

- [x] Basic RWKVv7 model implementation with `burn` framework  
- [x] Sequential token-by-token text generation  
- [x] CLI interface with configurable parameters  
- [x] Integration with `rwkv-tokenizer`  
- [ ] Investigate poor performance on larger models (e.g. 1.5B)
- [ ] Implement parallel generation mode  
- [ ] Improve support for additional backends (e.g. WGPU); Inference with WGPU seems numerically unstable and produces bad results

#### Mid-term Goals

- [ ] Improve inference speed
- [ ] Save and load model states (checkpointing)  
- [ ] Model quantization for faster inference and smaller memory footprint  
- [ ] Implement batch generation support
- [ ] Improve sampling strategies (temperature annealing, beam search)  

#### Long-term Goals

- [ ] Full training support (fine-tuning on custom datasets, weight initialization)  
- [ ] Multi-GPU and distributed inference support  
- [ ] Model export to ONNX and interoperability with other frameworks  
- [ ] Web-based interface and API service for model inference  

### Community and Contribution

Contributions and feedback are welcome!  
Feel free to open issues or submit pull requests for any feature requests or bug fixes.

---

_Last updated: 2025-05-26_
