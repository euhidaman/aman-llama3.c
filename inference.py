import torch
from model import Llama3
from params import ModelArgs, tokenizer, dtype_str
import argparse

def setup_args():
    parser = argparse.ArgumentParser(description='Run inference with Llama3 Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
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
    
    # Initialize model parameters
    params = ModelArgs()
    
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