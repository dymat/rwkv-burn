import torch
import numpy as np
import argparse
from safetensors.torch import safe_open, save_file

parser = argparse.ArgumentParser()
parser.add_argument("weights_path", help="Path to RWKVv7 pytorch weights.")
args = parser.parse_args()

if __name__ == '__main__':
    model = torch.load(args.weights_path, map_location="cpu")
    safetensors_weights = {key: value.cpu() for key, value in model.items()}
    save_file(safetensors_weights, f"{args.weights_path}.safetensors")