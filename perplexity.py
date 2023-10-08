import random
import math

# Sample text for training and evaluation
text = "This is a sample text for calculating perplexity. Perplexity measures how well a language model predicts sequences of characters. Lower perplexity is better."

# Split the data into training and evaluation sets (80% for training, 20% for evaluation)
random.seed(42)  # For reproducibility
split_index = int(0.8 * len(text))
train_data = text[:split_index]
eval_data = text[split_index:]

# Create a character-level language model
char_freq = {}
for char in train_data:
    char_freq[char] = char_freq.get(char, 0) + 1

# Function to calculate the likelihood of a sequence of characters
def calculate_likelihood(sequence):
    likelihood = 1.0
    vocab_size = len(char_freq)
    for char in sequence:
        # Laplace smoothing: add 1 to the count of all characters
        count = char_freq.get(char, 0) + 1
        total_count = len(char_freq) + (vocab_size * len(train_data))
        probability = count / total_count
        likelihood *= probability
    return likelihood

