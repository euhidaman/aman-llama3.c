#!/usr/bin/env python3

import os
import struct
import argparse
from typing import List, Tuple
import numpy as np


def read_binary_tokenizer(file_path: str) -> Tuple[int, List[Tuple[float, bytes]]]:
    """
    Read and parse the binary tokenizer file.
    Returns:
        Tuple containing:
        - max_token_length (int)
        - list of (score, token_bytes) tuples
    """
    tokens = []
    try:
        with open(file_path, 'rb') as f:
            # Read header
            max_token_length = struct.unpack('<i', f.read(4))[0]

            # Read tokens until EOF
            while True:
                try:
                    # Read score (float32) and length (int32)
                    score = struct.unpack('<f', f.read(4))[0]
                    length = struct.unpack('<i', f.read(4))[0]

                    # Read token data
                    token_bytes = f.read(length)

                    if not token_bytes:
                        break

                    tokens.append((score, token_bytes))
                except struct.error:
                    break  # End of file

        return max_token_length, tokens
    except Exception as e:
        raise RuntimeError(f"Error reading binary file: {str(e)}")


def verify_binary_structure(file_path: str) -> bool:
    """
    Verify the binary file structure and content.
    """
    try:
        # Read the binary file
        max_token_length, tokens = read_binary_tokenizer(file_path)

        print("\n=== Binary Tokenizer Verification ===")
        print(f"File: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"Max token length: {max_token_length}")
        print(f"Total tokens: {len(tokens)}")

        # Verify token count
        if len(tokens) != 512:
            print(f"ERROR: Expected 512 tokens, found {len(tokens)}")
            return False

        # Verify base tokens (0-255)
        print("\nChecking base tokens (0-255)...")
        base_tokens_correct = True
        for i in range(256):
            score, token = tokens[i]
            if len(token) != 1 or token[0] != i:
                print(f"ERROR: Base token {i} is incorrect")
                print(f"Expected: bytes([{i}])")
                print(f"Found: {token}")
                base_tokens_correct = False
                break

        if base_tokens_correct:
            print("✓ Base tokens verified successfully")

        # Check learned tokens
        print("\nChecking learned tokens (256-511)...")
        learned_tokens = tokens[256:]
        non_empty_learned = sum(1 for _, t in learned_tokens if len(t) > 0)
        print(f"Non-empty learned tokens: {non_empty_learned}")

        # Verify scores are monotonic for valid tokens
        print("\nVerifying token scores...")
        scores = [score for score, _ in tokens]
        base_scores = scores[:256]
        learned_scores = scores[256:]

        if base_scores != list(float(i) for i in range(256)):
            print("ERROR: Base token scores are not sequential")
            return False

        # Check if learned token scores are properly ordered
        prev_score = base_scores[-1]
        for i, score in enumerate(learned_scores):
            # Skip checking padding tokens (score=0)
            if score != 0 and score <= prev_score:
                print(f"ERROR: Non-monotonic score at position {i+256}")
                return False
            if score != 0:
                prev_score = score

        # Calculate and display statistics
        print("\n=== Token Statistics ===")
        token_lengths = [len(t) for _, t in tokens]
        print(f"Average token length: {np.mean(token_lengths):.2f} bytes")
        print(f"Min token length: {min(token_lengths)} bytes")
        print(f"Max token length: {max(token_lengths)} bytes")
        print(f"Total tokens size: {sum(token_lengths)} bytes")

        # Verify file size matches expected size
        expected_size = (
            4 +  # max_token_length (int32)
            # 8 bytes per token metadata + token data
            sum(8 + len(t) for _, t in tokens)
        )
        actual_size = os.path.getsize(file_path)

        print("\n=== File Size Verification ===")
        print(f"Expected size: {expected_size} bytes")
        print(f"Actual size: {actual_size} bytes")

        if expected_size != actual_size:
            print("ERROR: File size mismatch")
            return False

        print("\n=== Final Verdict ===")
        print("✓ Binary tokenizer file structure is valid")
        print("✓ All 512 tokens are present")
        print("✓ Base tokens (0-255) are correct")
        print("✓ File size matches expected structure")
        return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify binary tokenizer file structure and content')
    parser.add_argument(
        'file', help='Path to the binary tokenizer file (e.g., tokenizer_512_float32.bin)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return

    success = verify_binary_structure(args.file)
    if success:
        print("\n✅ Binary tokenizer verification passed!")
    else:
        print("\n❌ Binary tokenizer verification failed!")
        exit(1)


if __name__ == "__main__":
    main()
