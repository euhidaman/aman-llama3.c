import argparse
import array
import os
import glob
import json
import struct
import base64
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import tiktoken
import regex
from tiktoken.load import load_tiktoken_bpe

TOKENIZER_MODEL = "tokenizer.model"  # the llama tiktoken tokenizer model
DATA_CACHE_DIR = "data"


def load_tokenizer_model(model_path):
    """Load a tokenizer model from a file."""
    vocab = {}
    with open(model_path, 'r', encoding='utf-8') as f:
        for line in f:
            token_b64, rank = line.strip().split()
            token = base64.b64decode(token_b64)
            vocab[token] = int(rank)
    return vocab


class Tokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path

        # Load the vocabulary using our custom loader
        mergeable_ranks = load_tokenizer_model(model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        num_base_tokens = len(mergeable_ranks)
        num_reserved_special_tokens = 256

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_words = self.model.n_vocab
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(self, s: str, bos: bool, eos: bool, allowed_special=None, disallowed_special=()) -> List[int]:
        assert isinstance(s, str)
        if allowed_special is None:
            allowed_special = set()
        if disallowed_special is None:
            disallowed_special = set()
        t = self.model.encode(
            s,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        # Remove BOS and EOS tokens if they are present
        if t[0] == self.bos_id:
            t = t[1:]
        if t[-1] == self.eos_id:
            t = t[:-1]
        return self.model.decode(t)

    def export(self):
        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.model.decode_single_token_bytes(i)
            s = i
            tokens.append(t)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace(".model", ".bin")
        with open(tokenizer_bin, "wb") as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

    @staticmethod
    def train_vocab(corpus_dir: str, vocab_size: int):
        """Train a new vocabulary of the specified size on the given corpus directory"""
        print("Starting tokenizer training...")

        # Get first 10 shards as requested
        shard_files = sorted(
            glob.glob(os.path.join(corpus_dir, "*.json")))[:10]
        if not shard_files:
            raise FileNotFoundError(f"No JSON files found in {corpus_dir}")

        # Collect all text data
        all_text = []
        total_tokens = 0
        print("\nStep 1/4: Reading and collecting text from shards...")
        for filename in tqdm(shard_files, desc="Reading shards"):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    all_text.append(item["story"].strip())

        print(f"\nCollected {len(all_text):,} stories")

        # Convert text to bytes for training
        print("\nStep 2/4: Converting text to bytes...")
        byte_text = "\n".join(all_text).encode("utf-8")
        print(f"Total size of text: {len(byte_text)/1024/1024:.2f} MB")

        # Create base vocabulary (byte-level)
        base_vocab = {bytes([i]): i for i in range(256)}

        # Tokenize the text using regex pattern
        pattern = regex.compile(Tokenizer.pat_str)
        tokens = []

        print("\nStep 3/4: Tokenizing text and counting frequencies...")
        for text in tqdm(all_text, desc="Initial tokenization"):
            matches = pattern.finditer(text)
            for match in matches:
                token = match.group()
                if token:
                    tokens.append(token.encode("utf-8"))
                    total_tokens += 1

        print(f"Total tokens found: {total_tokens:,}")

        # Count token frequencies
        print("\nCounting unique tokens...")
        token_freqs = {}
        for token in tqdm(tokens, desc="Counting frequencies"):
            token_freqs[token] = token_freqs.get(token, 0) + 1

        # Sort by frequency
        print(f"Found {len(token_freqs):,} unique tokens")
        print("\nSorting tokens by frequency...")
        sorted_tokens = sorted(token_freqs.items(),
                               key=lambda x: x[1], reverse=True)

        # Take top vocab_size tokens
        print(f"\nStep 4/4: Creating final vocabulary of size {vocab_size}...")
        final_vocab = {token: i + 256 for i,
                       (token, freq) in enumerate(sorted_tokens[:vocab_size-256])}
        final_vocab.update(base_vocab)

        # Save the vocabulary
        model_path = f"tokenizer_{vocab_size}.model"
        print(f"\nSaving vocabulary to {model_path}...")
        with open(model_path, "w", encoding="utf-8") as f:
            for token, rank in tqdm(final_vocab.items(), desc="Writing vocabulary"):
                # Convert bytes to base64 string for storage
                token_b64 = base64.b64encode(token).decode('utf-8')
                f.write(f"{token_b64} {rank}\n")

        print(f"\nTraining completed!")
        print(f"Vocabulary size: {len(final_vocab):,} tokens")
        print(f"Most common token appears {sorted_tokens[0][1]:,} times")
        print(f"Least common token appears {sorted_tokens[-1][1]:,} times")
        print(f"Model saved to: {model_path}")


def process_shard(args):
    shard_id, shard, vocab_size = args
    tokenizer_model = f"tokenizer_{vocab_size}.model"
    try:
        # Create tokenizer with custom vocabulary loading
        enc = Tokenizer(tokenizer_model)

        with open(shard, "r") as f:
            data = json.load(f)

        all_tokens = []
        for example in tqdm(data, position=shard_id, desc=f"Processing shard {shard_id}"):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False, allowed_special=set(
            ), disallowed_special=set())  # encode the text, use BOS
            all_tokens.extend(tokens)

        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)

        # calculate the output filename
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

        # write the bytes
        os.makedirs(os.path.dirname(tokenized_filename), exist_ok=True)
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())

        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

    except Exception as e:
        print(f"Error processing shard {shard_id}: {str(e)}")
        raise


def pretokenize(vocab_size):
    """Pretokenize the dataset using the specified vocabulary size"""
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not shard_filenames:
        raise FileNotFoundError(
            f"No JSON files found in {data_dir}. Please make sure the dataset is downloaded and extracted.")

    # create output directory
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in parallel
    with ProcessPoolExecutor() as executor:
        args = [(i, shard, vocab_size)
                for i, shard in enumerate(shard_filenames)]
        list(executor.map(process_shard, args))

    print(f"Pretokenization completed. Tokenized files are saved in {bin_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train_vocab")
    train_parser.add_argument("--vocab_size", type=int, required=True)

    # Export command
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument(
        "-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer")

    # Pretokenize command
    pretok_parser = subparsers.add_parser("pretokenize")
    pretok_parser.add_argument(
        "--vocab_size", type=int, required=True, help="vocabulary size of the tokenizer")

    args = parser.parse_args()

    if args.command == "train_vocab":
        Tokenizer.train_vocab("data/TinyStories_all_data", args.vocab_size)
    elif args.command == "export":
        t = Tokenizer(args.tokenizer_model)
        t.export()
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size)
