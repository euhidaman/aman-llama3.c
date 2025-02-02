# params.py

import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass
from typing import Optional

# import the tokenizer we have created
from tokenizer import Tokenizer, load_tokenizer_model

# Initialize the tokenizer
tokenizer_model = "tokenizer_512.model"  # Path to your trained tokenizer model
tokenizer = Tokenizer(tokenizer_model)


@dataclass
class ModelArgs:
    dim: int = 128  # 4096
    n_layers: int = 12  # 32
    n_heads: int = 4  # 32
    n_kv_heads: Optional[int] = 1  # None
    vocab_size: int = tokenizer.n_words  # use the vocab size from the tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000  # 500000
    max_batch_size: int = 24
    max_seq_len: int = 512  # 8192 but their maximum chunk size when running inference is 2048
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate: float = 0.1


# Create an instance of ModelArgs
params = ModelArgs()
