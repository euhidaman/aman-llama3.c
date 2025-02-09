import struct
import numpy as np
from pathlib import Path
import os


def validate_model_binary(filepath):
    """Validate that the model binary file matches C code expectations"""
    try:
        with open(filepath, 'rb') as f:
            # 1. Read and validate Config struct
            config = {
                'dim': struct.unpack('i', f.read(4))[0],
                'n_layers': struct.unpack('i', f.read(4))[0],
                'n_heads': struct.unpack('i', f.read(4))[0],
                'n_kv_heads': struct.unpack('i', f.read(4))[0],
                'vocab_size': struct.unpack('i', f.read(4))[0],
                'seq_len': struct.unpack('i', f.read(4))[0],
                'norm_eps': struct.unpack('f', f.read(4))[0],
                'hidden_dim': struct.unpack('i', f.read(4))[0],
                'multiple_of': struct.unpack('i', f.read(4))[0],
                'rope_theta': struct.unpack('f', f.read(4))[0]
            }

            print("\nModel Configuration:")
            for k, v in config.items():
                print(f"{k}: {v}")

            # Validate parameters
            assert config['vocab_size'] == 512, f"vocab_size must be 512, got {config['vocab_size']}"
            assert 0 < config['dim'] <= 8192, f"Invalid dimension {config['dim']}"
            assert 0 < config['n_layers'] <= 100, f"Invalid n_layers {config['n_layers']}"
            assert 0 < config['n_heads'] <= 64, f"Invalid n_heads {config['n_heads']}"
            assert config['n_kv_heads'] <= config[
                'n_heads'], f"n_kv_heads {config['n_kv_heads']} > n_heads {config['n_heads']}"

            # Calculate expected file size
            expected_size = (
                10 * 4 +  # Config struct (10 fields * 4 bytes each)
                config['vocab_size'] * config['dim'] * 4 +  # token embeddings
                config['n_layers'] * (
                    config['dim'] * 4 +  # attention norm
                    config['dim'] * config['dim'] * 4 +  # wq
                    config['dim'] * config['dim'] * 4 +  # wk
                    config['dim'] * config['dim'] * 4 +  # wv
                    config['dim'] * config['dim'] * 4 +  # wo
                    config['dim'] * 4 +  # ffn norm
                    config['dim'] * config['hidden_dim'] * 4 +  # w1
                    config['hidden_dim'] * config['dim'] * 4 +  # w2
                    config['dim'] * config['hidden_dim'] * 4  # w3
                ) +
                config['dim'] * 4  # final norm
            )

            # Check file size
            f.seek(0, 2)  # Go to end of file
            actual_size = f.tell()

            print(f"\nFile size validation:")
            print(f"Expected: {expected_size:,} bytes")
            print(f"Actual: {actual_size:,} bytes")

            assert actual_size == expected_size, f"File size mismatch! Expected {expected_size:,} bytes but got {actual_size:,} bytes"

            print("\nModel binary validation successful! ✓")
            return True

    except Exception as e:
        print(f"\nError validating model binary: {str(e)}")
        return False


def validate_tokenizer_binary(filepath):
    """Validate that the tokenizer binary file matches C code expectations"""
    try:
        with open(filepath, 'rb') as f:
            # Read max token length
            max_token_length = struct.unpack('i', f.read(4))[0]
            print(f"\nTokenizer Configuration:")
            print(f"Max token length: {max_token_length}")

            # Read and validate all 512 tokens
            token_count = 0
            max_actual_length = 0

            while True:
                try:
                    # Read score and length
                    score = struct.unpack('f', f.read(4))[0]
                    length = struct.unpack('i', f.read(4))[0]

                    # Read token data
                    token_data = f.read(length)
                    max_actual_length = max(max_actual_length, length)

                    token_count += 1
                    if token_count > 512:
                        raise ValueError(
                            "Too many tokens in file (should be exactly 512)")

                except struct.error:
                    break

            print(f"Total tokens: {token_count}")
            print(f"Maximum token length in file: {max_actual_length}")

            # Validate token count
            assert token_count == 512, f"Expected 512 tokens, found {token_count}"

            # Validate max token length
            assert max_actual_length <= max_token_length, \
                f"Actual max token length {max_actual_length} exceeds header value {max_token_length}"

            print("\nTokenizer binary validation successful! ✓")
            return True

    except Exception as e:
        print(f"\nError validating tokenizer binary: {str(e)}")
        return False


def validate_exports(model_path, tokenizer_path):
    """Validate both model and tokenizer binaries"""
    print("\n=== Starting Binary Validation ===")

    model_ok = validate_model_binary(model_path)
    tokenizer_ok = validate_tokenizer_binary(tokenizer_path)

    if model_ok and tokenizer_ok:
        print("\n✓ All binary files validated successfully!")
        return True
    else:
        print("\n✗ Validation failed!")
        return False
