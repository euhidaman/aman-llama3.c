# model.py

import math
from typing import Optional, Tuple
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from params import ModelArgs, tokenizer, params


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(params.device).to(params.dtype)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads *
                            self.head_dim, bias=False).to(dtype=args.dtype)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False).to(dtype=args.dtype)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False).to(dtype=args.dtype)
        self.wo = nn.Linear(args.n_heads * self.head_dim,
                            args.dim, bias=False).to(dtype=args.dtype)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
            dtype=args.dtype,
            requires_grad=False
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
            dtype=args.dtype,
            requires_grad=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos is not None:
            # Handle kv caching for inference
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys, values = xk, xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / \
            math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).to(x.dtype)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False).to(dtype=params.dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False).to(dtype=params.dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False).to(dtype=params.dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout_rate = args.dropout_rate

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
        training=False,
    ):
        h = x + F.dropout(self.attention(self.attention_norm(x), freqs_cis,
                                         mask, start_pos), p=self.dropout_rate, training=training)
        out = h + F.dropout(self.feed_forward(self.ffn_norm(h)),
                            p=self.dropout_rate, training=training)
        return out


class Llama3(nn.Module):
    def __init__(self, params: ModelArgs, tokenizer):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_seq_len = params.max_seq_len
        self.tokenizer = tokenizer
        self.dtype = params.dtype

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim).to(dtype=self.dtype)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False).to(dtype=self.dtype)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        ).to(dtype=self.dtype)

        mask = torch.full(
            (params.max_seq_len, params.max_seq_len),
            float("-inf"),
            device=params.device,
            dtype=self.dtype
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor):
        bsz, seqlen = tokens.shape
        assert tokens.shape == targets.shape
        assert seqlen == self.max_seq_len

        h = self.tok_embeddings(tokens).to(dtype=self.dtype)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis, self.mask, start_pos=None, training=True)

        h = self.norm(h)
        logits = self.output(h)

        loss = self.criterion(
            logits.view(-1, self.vocab_size),
            targets.view(-1)
        )

        return logits, loss

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens).to(dtype=self.dtype)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = self.mask[:seqlen, :seqlen]
        mask = torch.hstack([
            torch.zeros((seqlen, start_pos),
                        device=tokens.device, dtype=self.dtype),
            mask
        ])

        for layer in self.layers:
            h = layer(h, freqs_cis, mask, start_pos=start_pos)

        h = self.norm(h)
        logits = self.output(h)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_gen_len: int = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = None,
    ) -> str:
        if max_gen_len is None:
            max_gen_len = self.max_seq_len - len(prompt)

        # Encode and prepare tokens
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor(tokens, dtype=torch.long,
                              device=self.params.device)
        tokens = tokens.unsqueeze(0)  # Add batch dimension [1, seq_len]

        start_pos = 0
        with torch.amp.autocast(device_type=self.params.device, dtype=self.dtype):
            for _ in range(max_gen_len):
                # Get logits for next token
                logits = self.forward_inference(
                    tokens[:, -self.max_seq_len:], start_pos)
                # Take the logits for the last token and convert to float32 for sampling
                logits = logits[:, -1, :].float()  # [1, vocab_size]

                if temperature > 0:
                    # Apply temperature
                    logits = logits / temperature
                    # Apply softmax to get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    # Sample next token
                    if top_p > 0:
                        next_token = self._sample_top_p_top_k(
                            probs, top_p, top_k)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Make sure next_token has shape [1, 1]
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 3:
                    next_token = next_token.squeeze(1)

                # Concatenate with previous tokens
                tokens = torch.cat([tokens, next_token], dim=1)
                start_pos += 1

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_id:
                    break

        # Decode the generated tokens
        generated_tokens = tokens[0].tolist()  # Remove batch dimension
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def _sample_top_p_top_k(self, probs, top_p, top_k=None):
        """Sample from the distribution with top-p and top-k filtering"""
        # Make sure we're working with 2D tensor [batch_size, vocab_size]
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = torch.zeros_like(probs).scatter_(-1, indices, values)

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(
                probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[...,
                                     1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for batch_idx in range(probs.size(0)):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                probs[batch_idx, indices_to_remove] = 0

        # Renormalize the probabilities
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs.div(probs_sum.clamp(min=1e-20))

        # Sample from the filtered distribution
        next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

        return next_token

    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration"""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_directory, "model.pth")
        torch.save(self.state_dict(), model_path)

        # Save configuration
        config = {
            "dim": self.params.dim,
            "n_layers": self.params.n_layers,
            "n_heads": self.params.n_heads,
            "n_kv_heads": self.params.n_kv_heads,
            "vocab_size": self.params.vocab_size,
            "multiple_of": self.params.multiple_of,
            "ffn_dim_multiplier": self.params.ffn_dim_multiplier,
            "norm_eps": self.params.norm_eps,
            "max_batch_size": self.params.max_batch_size,
            "max_seq_len": self.params.max_seq_len,
            "dropout_rate": self.params.dropout_rate,
            "rope_theta": self.params.rope_theta,
            "dtype": str(self.dtype)
        }

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_directory}")
        print(f"- Weights: {model_path}")
        print(f"- Config: {config_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str, device=None):
        """Load a pretrained model from a directory.

        Args:
            load_directory (str): Path to the directory containing the model files
            device (str, optional): Device to load the model to ('cuda' or 'cpu')
                                  If None, will use CUDA if available

        Returns:
            Llama3: A loaded model instance

        Raises:
            FileNotFoundError: If model files or tokenizer are not found
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        import os
        import json
        from tokenizer import Tokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Convert string dtype back to torch.dtype
        dtype_str = config.pop("dtype")
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }

        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}. "
                             f"Supported dtypes: {list(dtype_map.keys())}")

        dtype = dtype_map[dtype_str]

        # Create ModelArgs instance with loaded config
        from params import ModelArgs
        model_args = ModelArgs(device=device, dtype=dtype, **config)

        # Initialize tokenizer
        tokenizer_path = os.path.join(
            load_directory, f"tokenizer_{config['vocab_size']}_{dtype_str.split('.')[-1]}.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        tokenizer = Tokenizer(tokenizer_path)

        # Create model instance
        model = cls(model_args, tokenizer)

        # Load weights
        model_path = os.path.join(load_directory, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()

        return model

    def export_to_c(self, save_directory: str):
        """Export model weights to binary format for C inference"""
        import struct
        import numpy as np
        from pathlib import Path

        save_directory = Path(save_directory)
        bin_path = save_directory / "model.bin"

        # Define config that matches C struct exactly
        config = {
            'dim': self.params.dim,                    # 128
            'n_layers': self.params.n_layers,          # 12
            'n_heads': self.params.n_heads,            # 4
            # 4 (same as n_heads if not specified)
            'n_kv_heads': self.params.n_heads,
            'vocab_size': 512,                         # must be exactly 512
            'seq_len': self.params.max_seq_len,        # 512
            'norm_eps': self.params.norm_eps,          # 1e-5
            # 512 (4x the model dimension)
            'hidden_dim': self.params.dim * 4,
            'multiple_of': self.params.multiple_of,    # 256
            'rope_theta': self.params.rope_theta,      # 10000.0
        }

        print(f"\nExporting model to {bin_path}")
        print("\nConfiguration being exported:")
        for k, v in config.items():
            print(f"- {k}: {v}")

        with open(bin_path, 'wb') as f:
            # Write config struct - must match C exactly
            f.write(struct.pack('i', config['dim']))
            f.write(struct.pack('i', config['n_layers']))
            f.write(struct.pack('i', config['n_heads']))
            f.write(struct.pack('i', config['n_kv_heads']))
            f.write(struct.pack('i', config['vocab_size']))
            f.write(struct.pack('i', config['seq_len']))
            f.write(struct.pack('f', config['norm_eps']))
            f.write(struct.pack('i', config['hidden_dim']))
            f.write(struct.pack('i', config['multiple_of']))
            f.write(struct.pack('f', config['rope_theta']))

            # Get state dict
            state_dict = self.state_dict()

            # Write weights in the order expected by the C code
            def write_tensor(tensor):
                tensor = tensor.cpu().detach().numpy().astype(np.float32)
                tensor.tofile(f)

            # Token embeddings
            write_tensor(state_dict['tok_embeddings.weight'])

            # Layers
            for layer_id in range(config['n_layers']):
                prefix = f'layers.{layer_id}.'

                # Attention weights
                write_tensor(state_dict[prefix + 'attention_norm.weight'])
                write_tensor(state_dict[prefix + 'attention.wq.weight'])
                write_tensor(state_dict[prefix + 'attention.wk.weight'])
                write_tensor(state_dict[prefix + 'attention.wv.weight'])
                write_tensor(state_dict[prefix + 'attention.wo.weight'])

                # FFN weights
                write_tensor(state_dict[prefix + 'ffn_norm.weight'])
                write_tensor(state_dict[prefix + 'feed_forward.w1.weight'])
                write_tensor(state_dict[prefix + 'feed_forward.w2.weight'])
                write_tensor(state_dict[prefix + 'feed_forward.w3.weight'])

            # Final norm
            write_tensor(state_dict['norm.weight'])

        print(f"\nModel exported successfully to {bin_path}")
        print(f"File size: {bin_path.stat().st_size / 1024 / 1024:.2f} MB")

    def get_model_size(self):
        """Calculate and return model size in millions of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params / 1e6

    def print_model_size(self):
        """Print model size information"""
        total_params = self.get_model_size()
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad) / 1e6

        print("\nModel Size Information:")
        print(f"- Total Parameters: {total_params:.2f}M")
        print(f"- Trainable Parameters: {trainable_params:.2f}M")
        print(f"- Model Dtype: {self.dtype}")
        print(f"- Device: {next(self.parameters()).device}")
