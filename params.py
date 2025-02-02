# params.py

from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from torch import nn
import argparse
from dataclasses import dataclass
from typing import Optional


def get_dtype_from_args():
    """Get dtype from command line arguments"""
    parser = argparse.ArgumentParser(description='Model configuration')
    parser.add_argument('--dtype', type=str,
                        choices=['float16', 'bfloat16', 'float32'],
                        default='float32',
                        help='Data type to use (default: float32)')
    # Parse known args only to avoid conflicts with other scripts
    args, _ = parser.parse_known_args()
    return args.dtype


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype"""
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


# Get dtype from command line args
dtype_str = get_dtype_from_args()
print(f"Using dtype: {dtype_str}")

# import the tokenizer we have created

# Initialize the tokenizer with dtype-specific model
tokenizer_model = f"tokenizer_512_{dtype_str}.model"
try:
    print(f"Loading tokenizer model: {tokenizer_model}")
    tokenizer = Tokenizer(tokenizer_model)
except FileNotFoundError:
    print(f"\nError: Tokenizer model '{tokenizer_model}' not found!")
    print(f"Please run first:")
    print(
        f"python tokenizer.py train_vocab --vocab_size=512 --dtype={dtype_str}")
    print(
        f"python tokenizer.py pretokenize --vocab_size=512 --dtype={dtype_str}")
    raise


@dataclass
class ModelArgs:
    dim: int = 128  # 4096
    n_layers: int = 12  # 32
    n_heads: int = 4  # 32
    n_kv_heads: Optional[int] = None  # Changed from 1 to None for consistency
    vocab_size: int = None  # Will be set after tokenizer initialization
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_batch_size: int = 24
    max_seq_len: int = 512
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate: float = 0.1
    dtype: torch.dtype = get_torch_dtype(dtype_str)

    def __post_init__(self):
        if self.vocab_size is None:
            self.vocab_size = tokenizer.n_words
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads


# Create an instance of ModelArgs
params = ModelArgs()

# Print configuration
if __name__ == "__main__":
    print("\nModel Configuration:")
    print("-" * 50)
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")
