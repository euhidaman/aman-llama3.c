# Example script to demonstrate tokenization
from tokenizer import Tokenizer

# Initialize the tokenizer
tokenizer_model = "tokenizer_512.model"  # Path to your trained tokenizer model
tokenizer = Tokenizer(tokenizer_model)

# Define a sample text
sample_text = "Once upon a time, in a faraway land, there lived a wise old owl."

# Tokenize the sample text
tokens = tokenizer.encode(sample_text, bos=True, eos=True)

# Print the tokens
print("Tokenized text:", tokens)

# Decode the tokens back to text
decoded_text = tokenizer.decode(tokens)
print("Decoded text:", decoded_text)
