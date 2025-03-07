# params.py

from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from torch import nn
import argparse
from dataclasses import dataclass
from typing import Optional
import os


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


def validate_tokenizer_config(tokenizer_path: str) -> bool:
    """Validate that tokenizer has exactly 512 tokens"""
    try:
        tokenizer = Tokenizer(tokenizer_path)
        if tokenizer.n_words != 512:
            print(f"\nError: Invalid vocabulary size")
            print(f"Expected: 512 tokens")
            print(f"Found: {tokenizer.n_words} tokens")
            return False
        return True
    except Exception as e:
        print(f"Error validating tokenizer: {str(e)}")
        return False


# Get dtype from command line args
dtype_str = get_dtype_from_args()
print(f"Using dtype: {dtype_str}")

# Initialize the tokenizer with dtype-specific model
tokenizer_model = f"tokenizer_512_{dtype_str}.model"
tokenizer_bin = tokenizer_model.replace('.model', '.bin')

# Check if both model and binary files exist
if not os.path.exists(tokenizer_model) or not os.path.exists(tokenizer_bin):
    print(f"\nError: Tokenizer files not found!")
    print(f"Missing files:")
    if not os.path.exists(tokenizer_model):
        print(f"- {tokenizer_model}")
    if not os.path.exists(tokenizer_bin):
        print(f"- {tokenizer_bin}")
    print("\nPlease run first:")
    print(
        f"python tokenizer.py train_vocab --vocab_size=512 --dtype={dtype_str}")
    print(
        f"python tokenizer.py pretokenize --vocab_size=512 --dtype={dtype_str}")
    raise FileNotFoundError("Tokenizer files not found")

# Load and validate tokenizer
try:
    print(f"Loading tokenizer model: {tokenizer_model}")
    tokenizer = Tokenizer(tokenizer_model)
    if not validate_tokenizer_config(tokenizer_model):
        raise ValueError("Tokenizer validation failed")
except Exception as e:
    print(f"\nError: Failed to load or validate tokenizer!")
    print(f"Error details: {str(e)}")
    print("\nPlease ensure the tokenizer was properly trained with:")
    print(
        f"python tokenizer.py train_vocab --vocab_size=512 --dtype={dtype_str}")
    raise


@dataclass
class ModelArgs:
    dim: int = 256                # Model dimension
    n_layers: int = 12           # Number of layers
    n_heads: int = 4             # Number of attention heads
    n_kv_heads: Optional[int] = None
    vocab_size: int = 512        # Must be exactly 512
    multiple_of: int = 256       # Dimension must be divisible by this
    ffn_dim_multiplier: Optional[float] = None
    max_seq_len: int = 512
    max_batch_size: int = 24
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32
    dropout_rate: float = 0.0    # Add dropout rate parameter

    def __post_init__(self):
        # Strict validation for vocab_size
        if self.vocab_size != 512:
            raise ValueError(
                f"vocab_size must be exactly 512, got {self.vocab_size}")

        # Ensure n_kv_heads is set properly
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Validate dimensions
        if self.dim % self.multiple_of != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by multiple_of ({self.multiple_of})")

        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")


# Create an instance of ModelArgs
params = ModelArgs()

# Print configuration
if __name__ == "__main__":
    print("\nModel Configuration:")
    print("-" * 50)
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")
