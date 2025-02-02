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
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='initial learning rate (default: 1e-2)')
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


def get_batch(data, split, batch_size, max_seq_len, device):
    """Generate a batch of data for training or validation"""
    data_split = data[:int(0.9 * len(data))
                      ] if split == 'train' else data[int(0.9 * len(data)):]
    ix = torch.randint(len(data_split) - max_seq_len, (batch_size,))

    # Use clone() instead of copy() for PyTorch tensors
    x = torch.stack([data_split[i:i + max_seq_len].clone() for i in ix])
    y = torch.stack([data_split[i + 1:i + max_seq_len + 1].clone()
                    for i in ix])

    return x.to(device), y.to(device)


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
    """Save a training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': iteration,
        'loss': loss,
        'dtype': str(model.dtype),
        'params': asdict(model.params)
    }

    # Convert dtype in params to string
    if 'dtype' in checkpoint['params']:
        checkpoint['params']['dtype'] = str(checkpoint['params']['dtype'])

    checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")


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

    # Setup data directory and load data
    data_dir = get_data_dir(args.dtype)

    # Determine max_files based on test_run flag
    max_files = 2 if args.test_run else (
        None if args.max_files == -1 else args.max_files)

    try:
        # Print memory info before loading
        process = psutil.Process()
        print("\nInitial system memory:",
              f"{psutil.virtual_memory().available/1024/1024/1024:.2f} GB")

        # Load data with progress
        data = load_preprocessed_data(data_dir, max_files)
        # Convert numpy array to tensor
        data = torch.from_numpy(data).to(torch.long)

        # Print memory info after loading
        print("Memory usage after data loading:",
              f"{process.memory_info().rss/1024/1024/1024:.2f} GB")

        # Initialize model parameters
        params = ModelArgs()
        print("\nModel Configuration:")
        print("-" * 50)
        for key, value in asdict(params).items():
            print(f"{key}: {value}")

        # Initialize the model
        model = Llama3(params, tokenizer).to(params.device)
        print(
            f"\nModel has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        scheduler = create_lr_scheduler(
            optimizer,
            args.warmup_iters,
            args.max_iters,
            args.min_lr,
            args.lr
        )

        # Create save directory
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

        progress_bar = tqdm(range(args.max_iters), desc="Training")
        for iter in progress_bar:
            # Get batch and train
            xb, yb = get_batch(data, 'train', args.batch_size,
                               params.max_seq_len, params.device)
            logits, loss = model(xb, targets=yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Evaluation
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(model, data, args.batch_size)
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
                                iter, loss.item(), save_dir)

        # Save final model
        final_path = save_dir / 'final_model.pt'
        torch.save(model.state_dict(), final_path)
        print(f"\nTraining completed! Final model saved to {final_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
