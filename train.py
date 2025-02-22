# train.py

from model import Llama3
from params import ModelArgs, tokenizer, dtype_str
import sys
import os
import glob
import torch
import math
import time
import json
import numpy as np
from dataclasses import asdict
import argparse
from datetime import datetime
from pathlib import Path
import psutil
from tqdm import tqdm
import warnings
import struct
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

# Suppress complex values warning
warnings.filterwarnings(
    "ignore", message="Casting complex values to real discards the imaginary part")

# importing the model config

# importing the model


def setup_args():
    parser = argparse.ArgumentParser(description='Train Llama3 Model')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size for training (default: 24)')
    parser.add_argument('--max_iters', type=int, default=2000,
                        help='number of iterations to train for (default: 2000)')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='how often to evaluate the model (default: 100)')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='how often to save checkpoints (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--warmup_iters', type=int, default=50,
                        help='number of warmup iterations (default: 50)')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32'],
                        default=dtype_str, help=f'data type to use (default: {dtype_str})')
    parser.add_argument('--max_files', type=int, default=5,
                        help='maximum number of files to load (default: 5, use -1 for all files)')
    parser.add_argument('--test_run', action='store_true',
                        help='run with minimal data for testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients (default: 4)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    return parser.parse_args()


def get_data_dir(dtype_str):
    """Get the directory containing tokenized data for the specified dtype"""
    return os.path.join("data", f"tok512_{dtype_str}")


def load_data_in_chunks(data_files, chunk_size=1000000):
    """Load data in chunks to manage memory usage"""
    print("\nUsing chunk-based loading...")

    total_tokens = 0
    for file in tqdm(data_files, desc="Counting tokens"):
        total_tokens += os.path.getsize(file) // 2

    final_data = np.zeros(total_tokens, dtype=np.uint16)
    current_pos = 0

    for file in tqdm(data_files, desc="Loading chunks"):
        try:
            with open(file, "rb") as f:
                while True:
                    chunk = np.fromfile(f, dtype=np.uint16, count=chunk_size)
                    if chunk.size == 0:
                        break
                    final_data[current_pos:current_pos + chunk.size] = chunk
                    current_pos += chunk.size
        except Exception as e:
            print(f"\nError loading chunk from {file}: {str(e)}")
            continue

    print(f"\nChunk loading completed!")
    print(f"Total tokens loaded: {current_pos:,}")
    return final_data[:current_pos]


def load_preprocessed_data(data_dir, max_files=None):
    """Load preprocessed data from binary files with memory-efficient approach"""
    print(f"\nLoading data from {data_dir}")
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))

    if not data_files:
        raise FileNotFoundError(
            f"No .bin files found in {data_dir}. "
            f"Please run pretokenize with --dtype={dtype_str} first."
        )

    if max_files is not None:
        data_files = data_files[:max_files]
        print(f"Loading first {max_files} files for testing...")

    total_size = sum(os.path.getsize(f) for f in data_files)
    required_memory = total_size * 2
    available_memory = psutil.virtual_memory().available

    print(f"\nMemory Analysis:")
    print(f"Total data size: {total_size/1024/1024/1024:.2f} GB")
    print(
        f"Estimated memory required: {required_memory/1024/1024/1024:.2f} GB")
    print(f"Available system memory: {available_memory/1024/1024/1024:.2f} GB")

    if required_memory > available_memory:
        print("\nWarning: Required memory exceeds available system memory!")
        print("Switching to chunk-based loading...")
        return load_data_in_chunks(data_files)

    try:
        all_data = []
        total_tokens = 0

        for file in tqdm(data_files, desc="Loading data files"):
            try:
                with open(file, "rb") as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    all_data.append(data)
                    total_tokens += len(data)

                    if len(all_data) % 10 == 0:
                        current_memory = psutil.Process().memory_info().rss
                        print(f"\nLoaded {len(all_data)} files, "
                              f"Current memory usage: {current_memory/1024/1024/1024:.2f} GB")

            except Exception as e:
                print(f"\nError loading file {file}: {str(e)}")
                continue

        print(f"\nSuccessfully loaded {len(data_files)} files")
        print(f"Total tokens: {total_tokens:,}")

        print("\nConcatenating data arrays...")
        final_data = np.concatenate(all_data)
        print(f"Final data shape: {final_data.shape}")

        return final_data

    except MemoryError:
        print("\nMemoryError encountered during loading!")
        print("Switching to chunk-based loading...")
        return load_data_in_chunks(data_files)


def get_batch(data, split, batch_size, max_seq_len, device, dtype=None):
    """Generate a batch of data for training or validation"""
    data_split = data[:int(0.9 * len(data))
                      ] if split == 'train' else data[int(0.9 * len(data)):]
    ix = torch.randint(len(data_split) - max_seq_len, (batch_size,))

    # Keep input tokens as long for embedding layer
    x = torch.stack([data_split[i:i + max_seq_len].clone() for i in ix])
    y = torch.stack([data_split[i + 1:i + max_seq_len + 1].clone()
                    for i in ix])

    return x.to(device).long(), y.to(device).long()


def serialize_fp32(file, tensor):
    """Writes tensor to file in fp32 format"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def export_model_to_c(model, filepath):
    """Export model to C binary format regardless of configuration"""
    try:
        with open(filepath, 'wb') as f:
            # Write actual config values - allow any values
            config = struct.pack(
                'iiiiiifiif',
                model.args.dim,          # dim
                model.args.n_layers,     # n_layers
                model.args.n_heads,      # n_heads
                model.args.n_kv_heads,   # n_kv_heads
                model.args.vocab_size,   # vocab_size
                model.args.max_seq_len,  # seq_len
                model.args.norm_eps,     # norm_eps
                model.hidden_dim,        # hidden_dim
                model.args.multiple_of,  # multiple_of
                model.args.rope_theta    # rope_theta
            )
            f.write(config)

            # Write all weights
            serialize_fp32(f, model.tok_embeddings.weight)
            for layer in model.layers:
                serialize_fp32(f, layer.attention_norm.weight)
                serialize_fp32(f, layer.attention.wq.weight)
                serialize_fp32(f, layer.attention.wk.weight)
                serialize_fp32(f, layer.attention.wv.weight)
                serialize_fp32(f, layer.attention.wo.weight)
                serialize_fp32(f, layer.ffn_norm.weight)
                serialize_fp32(f, layer.feed_forward.w1.weight)
                serialize_fp32(f, layer.feed_forward.w2.weight)
                serialize_fp32(f, layer.feed_forward.w3.weight)
            serialize_fp32(f, model.norm.weight)

            if not torch.equal(model.tok_embeddings.weight, model.output.weight):
                serialize_fp32(f, model.output.weight)

            return True
    except Exception as e:
        print(f"\nError during export: {e}")
        return False


@torch.no_grad()
def estimate_loss(model, data, batch_size, eval_iters=5):
    """Estimate loss on train and validation splits"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, batch_size,
                             model.max_seq_len, model.params.device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=model.params.dtype == torch.float16):
                logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def create_lr_scheduler(optimizer, warmup_iters, max_iters, min_lr, max_lr):
    """Create learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / warmup_iters
        else:
            decay_ratio = (current_iter - warmup_iters) / \
                (max_iters - warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return max(min_lr / max_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, iteration, loss, save_dir):
    """Save checkpoint and export model regardless of validation"""
    # Save PyTorch checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iteration': iteration,
        'loss': loss,
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Export C-compatible binary
    binary_path = os.path.join(save_dir, 'model.bin')
    export_success = export_model_to_c(model, binary_path)

    # Always validate and show configuration/errors, but don't prevent saving
    if export_success:
        print("\nValidating exported binaries...")
        try:
            with open(binary_path, 'rb') as f:
                config = struct.unpack('iiiiiifiif', f.read(40))

            print("\nModel Configuration:")
            print(f"dim: {config[0]}")
            print(f"n_layers: {config[1]}")
            print(f"n_heads: {config[2]}")
            print(f"n_kv_heads: {config[3]}")
            print(f"vocab_size: {config[4]}")
            print(f"seq_len: {config[5]}")
            print(f"norm_eps: {config[6]}")
            print(f"hidden_dim: {config[7]}")
            print(f"multiple_of: {config[8]}")
            print(f"rope_theta: {config[9]}")

            if config[4] != 512:  # vocab_size check
                print(f"\nWarning: vocab_size must be 512, got {config[4]}")

            print(
                "\nModel binary exported successfully (with non-standard configuration)")
            print(f"Binary saved to: {binary_path}")

        except Exception as e:
            print(
                f"\nWarning: Binary validation failed but file was saved: {e}")
    else:
        print("\nError: Failed to export model binary")


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


def save_config(save_dir, params, args):
    """Save configuration with proper dtype handling"""
    config_path = save_dir / 'config.json'

    # Convert ModelArgs to dict and handle dtype
    model_params = asdict(params)
    model_params['dtype'] = str(model_params['dtype'])

    # Convert training args to dict
    training_args = vars(args)

    config = {
        'model_params': model_params,
        'training_args': training_args
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_path}")


def main():
    # Parse arguments
    args = setup_args()

    # Set the dtype based on args
    if args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Setup data directory and load data
    data_dir = get_data_dir(args.dtype)

    try:
        # Print memory info before loading
        process = psutil.Process()
        print("\nInitial system memory:",
              f"{psutil.virtual_memory().available/1024/1024/1024:.2f} GB")

        # Load data with progress
        data = load_preprocessed_data(data_dir, args.max_files)
        data = torch.from_numpy(data).to(torch.long)

        # Print memory info after loading
        print("Memory usage after data loading:",
              f"{process.memory_info().rss/1024/1024/1024:.2f} GB")

        # Initialize model parameters
        params = ModelArgs()
        params.dtype = dtype

        print("\nModel Configuration:")
        print("-" * 50)
        for key, value in asdict(params).items():
            print(f"{key}: {value}")

        # Initialize model and move to device
        model = Llama3(params, tokenizer).to(params.device)
        model = model.float()  # Convert to float32 for stability

        print(
            f"\nModel has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8  # Added eps for numerical stability
        )

        scheduler = create_lr_scheduler(
            optimizer,
            args.warmup_iters,
            args.max_iters,
            args.min_lr,
            args.lr
        )

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler(enabled=args.dtype == 'float16')

        # Create save directory with timestamp
        save_dir = Path(
            f'models/llama3_{args.dtype}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        try:
            save_config(save_dir, params, args)
        except Exception as e:
            print(f"Warning: Error saving config: {str(e)}")
            print("Continuing with training...")

        # Training loop
        print("\nStarting training...")
        start_time = time.time()
        best_val_loss = float('inf')
        accumulated_loss = 0

        progress_bar = tqdm(range(args.max_iters), desc="Training")
        for iter in progress_bar:
            try:
                # Memory management at start of iteration
                if iter % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Zero gradients at start of accumulation
                if iter % args.gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = 0

                # Get batch and train
                xb, yb = get_batch(data, 'train', args.batch_size,
                                   params.max_seq_len, params.device)

                # Forward pass with autocast
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=args.dtype == 'float16'):
                    logits, loss = model(xb, targets=yb)
                    loss = loss / args.gradient_accumulation_steps
                    accumulated_loss += loss.item()

                # Backward pass
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step optimizer after accumulation
                if (iter + 1) % args.gradient_accumulation_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                        optimizer.step()

                    scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

                # Evaluation
                if iter % args.eval_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        losses = estimate_loss(model, data, args.batch_size)
                    model.train()

                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time

                    print(f"\niter {iter:,}/{args.max_iters:,} | "
                          f"lr {current_lr:.2e} | "
                          f"train loss {losses['train']:.4f} | "
                          f"val loss {losses['val']:.4f} | "
                          f"elapsed {elapsed:.2f}s")

                    # Save best model
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        save_checkpoint(model, optimizer, scheduler,
                                        iter, losses['val'], save_dir)

                # Regular checkpoint saving
                if iter % args.save_interval == 0 and iter > 0:
                    save_checkpoint(model, optimizer, scheduler,
                                    iter, accumulated_loss, save_dir)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"\nOOM error in iteration {iter}. Trying to recover...")
                    continue
                else:
                    raise e

        # Save final model
        # Save final model in PyTorch format
        final_path = save_dir / 'final_model.pt'
        torch.save(model.state_dict(), final_path)

        # Export model to binary format for C inference
        binary_path = save_dir / 'model.bin'
        # Use version 1 format
        export_model_to_c(model, binary_path, version=1)

        print("\nValidating exported binaries...")
        validation_success = validate_model_binary(binary_path)

        if validation_success:
            print(f"\nTraining completed!")
            print(f"Models saved and validated:")
            print(f"- PyTorch format: {final_path}")
            print(f"- C binary format: {binary_path}")
            print(f"Best validation loss: {best_val_loss:.4f}")
        else:
            print("\nWarning: Binary validation failed! Please check the exported files.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Handle CUDA memory settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Print CUDA information
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(
            f"Available CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Set autocast dtype
        torch.set_float32_matmul_precision('high')
    else:
        print("CUDA is not available. Training will be slow on CPU.")

    try:
        main()
    except Exception as e:
        print(f"\nFatal error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
