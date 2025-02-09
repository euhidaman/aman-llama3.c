import argparse
from validate import validate_exports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate model and tokenizer binary files')
    parser.add_argument('--model', required=True,
                        help='Path to model.bin file')
    parser.add_argument('--tokenizer', required=True,
                        help='Path to tokenizer.bin file')

    args = parser.parse_args()

    validate_exports(args.model, args.tokenizer)
