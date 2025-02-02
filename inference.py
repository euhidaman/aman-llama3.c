import torch
from model import Llama3
from params import ModelArgs, get_torch_dtype
import argparse
import os


def setup_args():
    parser = argparse.ArgumentParser(
        description='Run inference with Llama3 Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--dtype', type=str,
                        choices=['float16', 'bfloat16', 'float32'],
                        default='float16',  # Changed default to float16
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

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Using dtype: {args.dtype}")

    # Set up tokenizer
    tokenizer_model = f"tokenizer_512_{args.dtype}.model"
    try:
        from tokenizer import Tokenizer
        tokenizer = Tokenizer(tokenizer_model)
    except FileNotFoundError:
        print(f"\nError: Tokenizer model '{tokenizer_model}' not found!")
        print(f"Please run first:")
        print(
            f"python tokenizer.py train_vocab --vocab_size=512 --dtype={args.dtype}")
        return

    # Initialize model parameters with specified dtype
    params = ModelArgs()
    params.dtype = get_torch_dtype(args.dtype)
    params.device = device

    # Initialize model
    model = Llama3(params, tokenizer)

    # Load checkpoint
    try:
        checkpoint = torch.load(
            args.model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        return

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    print(f"\nGenerating text with:")
    print(f"- Temperature: {args.temperature}")
    print(f"- Top-p: {args.top_p}")
    print(f"- Max length: {args.max_gen_len}")

    try:
        generated_text = model.generate(
            prompt=args.prompt,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p
        )

        print("\nPrompt:", args.prompt)
        print("\nGenerated text:")
        print(generated_text)
    except Exception as e:
        print(f"\nError during text generation: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
