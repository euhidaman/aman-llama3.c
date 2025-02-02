# Example script to demonstrate tokenization with different dtypes
import argparse
from tokenizer import Tokenizer


def check_tokenization(dtype='float32'):
    """
    Test tokenization with specified dtype
    Args:
        dtype (str): Data type to use ('float16', 'bfloat16', or 'float32')
    """
    # Initialize the tokenizer with dtype-specific model
    # Path to your trained tokenizer model
    tokenizer_model = f"tokenizer_512_{dtype}.model"
    print(f"\nUsing tokenizer model: {tokenizer_model}")
    tokenizer = Tokenizer(tokenizer_model)

    # Define sample texts
    sample_texts = [
        "Once upon a time, in a faraway land, there lived a wise old owl.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of the tokenizer.",
    ]

    print(f"\nTesting tokenization with dtype: {dtype}")
    print("-" * 50)

    # Process each sample text
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}:")
        print("Original text:", text)

        # Tokenize the sample text
        tokens = tokenizer.encode(text, bos=True, eos=True)
        print("Tokenized:", tokens)

        # Decode the tokens back to text
        decoded_text = tokenizer.decode(tokens)
        print("Decoded text:", decoded_text)

        # Print some statistics
        print(f"Number of tokens: {len(tokens)}")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test tokenizer with different dtypes')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32'],
                        default='float32', help='Data type to use (default: float32)')

    args = parser.parse_args()

    try:
        check_tokenization(args.dtype)
    except FileNotFoundError as e:
        print(f"\nError: Could not find tokenizer model file.")
        print(f"Please make sure you have trained the tokenizer with the specified dtype first:")
        print(
            f"python tokenizer.py train_vocab --vocab_size=512 --dtype={args.dtype}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
