import torch
from model import Llama3
from params import ModelArgs, tokenizer, get_torch_dtype
import argparse


def setup_args():
    parser = argparse.ArgumentParser(
        description='Run inference with Llama3 Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--dtype', type=str,
                        choices=['float16', 'bfloat16', 'float32'],
                        default='float16',
                        help='Data type to use')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='Text prompt to start generation')
    parser.add_argument('--max_gen_len', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for sampling (0.0 = greedy, higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling threshold')
    return parser.parse_args()


def main():
    args = setup_args()

    # Set up tokenizer for the specified dtype
    tokenizer_model = f"tokenizer_512_{args.dtype}.model"
    try:
        from tokenizer import Tokenizer
        tokenizer = Tokenizer(tokenizer_model)
    except FileNotFoundError:
        print(f"\nError: Tokenizer model '{tokenizer_model}' not found!")
        print(f"Please run first:")
        print(
            f"python tokenizer.py train_vocab --vocab_size=512 --dtype={args.dtype}")
        print(
            f"python tokenizer.py pretokenize --vocab_size=512 --dtype={args.dtype}")
        return

    # Initialize model parameters with specified dtype
    params = ModelArgs()
    params.dtype = get_torch_dtype(args.dtype)

    # Initialize model
    model = Llama3(params, tokenizer)

    # Load the trained weights
    checkpoint = torch.load(args.model_path, map_location=params.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to device and set to evaluation mode
    model = model.to(params.device)
    model.eval()

    # Generate text
    generated_text = model.generate(
        prompt=args.prompt,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("\nPrompt:", args.prompt)
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
