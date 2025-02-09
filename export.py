"""
Export script to convert .model files to .bin files for llama-type models and tokenizers.
Supports both model weights and tokenizer conversion.
"""

import os
import struct
import argparse
import json
import numpy as np
import torch
from torch import nn

def serialize_fp32(file, tensor):
    """Writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d) 
    file.write(b)

def serialize_int8(file, tensor):
    """Writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size=64):
    """
    Quantizes a tensor to Q8_0 format (symmetric int8, range [-127,127])
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()
    w = w.reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    quant = w / scale[:,None]
    int8val = torch.round(quant).to(torch.int8)
    return int8val, scale

def export_model(model, filepath, version=1):
    """
    Exports model weights to .bin format
    Versions:
    - v1: Full float32 weights
    - v2: Int8 quantized weights
    """
    out_file = open(filepath, 'wb')
    
    # Write header (256 bytes)
    out_file.write(struct.pack('I', 0x616b3432)) # Magic number
    out_file.write(struct.pack('i', version))     # Version
    
    # Write model config
    config = model.config
    header = struct.pack('iiiiiii', 
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads if hasattr(config, 'n_kv_heads') else config.n_heads,
        config.vocab_size,
        config.max_seq_len
    )
    out_file.write(header)
    
    # Write other flags
    shared_weights = hasattr(model, 'shared_weights') and model.shared_weights
    out_file.write(struct.pack('B', int(shared_weights)))
    
    # Pad remaining header
    pad = 256 - out_file.tell()
    out_file.write(b'\0' * pad)
    
    # Write weights
    if version == 1:
        # Full float32
        for name, param in model.named_parameters():
            serialize_fp32(out_file, param)
            
    elif version == 2:
        # Int8 quantized
        for name, param in model.named_parameters():
            if param.ndim == 2: # Only quantize matrices
                q, s = quantize_q80(param)
                serialize_int8(out_file, q)
                serialize_fp32(out_file, s)
            else:
                serialize_fp32(out_file, param)
                
    out_file.close()
    print(f"Exported model to {filepath}")

def export_tokenizer(tokenizer, filepath):
    """
    Exports tokenizer to binary format
    """
    out_file = open(filepath, 'wb')
    
    # Write vocab size
    vocab_size = len(tokenizer.vocab)
    out_file.write(struct.pack('i', vocab_size))
    
    # Write tokens and scores
    for token, score in tokenizer.vocab.items():
        # Write token length and bytes
        token_bytes = token.encode('utf-8')
        out_file.write(struct.pack('i', len(token_bytes)))
        out_file.write(token_bytes)
        # Write score
        out_file.write(struct.pack('f', float(score)))
        
    out_file.close()
    print(f"Exported tokenizer to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .model file path")
    parser.add_argument("output", help="Output .bin file path") 
    parser.add_argument("--type", choices=['model', 'tokenizer'], required=True,
                      help="Type of file to convert")
    parser.add_argument("--version", type=int, default=1,
                      help="Version of export format (1=float32, 2=int8)")
    args = parser.parse_args()
    
    if args.type == 'model':
        # Load model
        model = torch.load(args.input)
        export_model(model, args.output, args.version)
    else:
        # Load tokenizer 
        with open(args.input) as f:
            tokenizer = json.load(f)
        export_tokenizer(tokenizer, args.output)