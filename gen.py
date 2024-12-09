#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to Generate Vocabulary from a Large Corpus and Update a Model JSON. You can alter this in any way.
"""

import json
import os
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def generate_vocabulary_from_corpus(corpus: str, vocab_size: int) -> dict:
    """
    Generate a vocabulary and token-to-ID mapping from the given text corpus.
    """
    if not corpus.strip():
        raise ValueError("Input corpus is empty or contains no valid text.")

    # Tokenize the corpus
    tokens = word_tokenize(corpus.lower())
    
    if not tokens:
        raise ValueError("Tokenization produced no tokens. Ensure the input corpus contains valid text.")

    # Count token frequencies
    token_freq = Counter(tokens)
    
    # Select the most common tokens based on vocab_size
    most_common_tokens = [token for token, _ in token_freq.most_common(vocab_size)]
    
    # Create token-to-ID mapping
    token_to_id = {token: idx for idx, token in enumerate(most_common_tokens)}
    return token_to_id

def update_model_with_vocabulary(json_file: str, token_to_id: dict):
    """
    Update the model JSON file with a `token_to_id` mapping and ensure embeddings match the vocabulary.
    """
    # Load the model JSON file
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Model JSON file not found: {json_file}")
    
    with open(json_file, "r") as f:
        model_dict = json.load(f)
    
    # Update the model with token mappings
    model_dict["token_to_id"] = token_to_id
    vocab_size = len(token_to_id)
    
    # Ensure embeddings match the new vocabulary size
    if "embeddings" in model_dict:
        embed_dim = len(model_dict["embeddings"][0])
        model_dict["embeddings"] = (np.random.randn(vocab_size, embed_dim) * 0.01).tolist()
    else:
        raise ValueError("Embeddings are missing in the model JSON.")

    # Save the updated model JSON file
    with open(json_file, "w") as f:
        json.dump(model_dict, f, indent=4)
    
    print(f"Model updated successfully with vocabulary. Saved to: {json_file}")

def validate_input_text(text: str, token_to_id: dict):
    """
    Validate that the input text contains tokens present in the vocabulary.
    """
    tokens = word_tokenize(text.lower())
    valid_tokens = [token for token in tokens if token in token_to_id]
    
    if not valid_tokens:
        missing_tokens = [token for token in tokens if token not in token_to_id]
        raise ValueError(
            f"Input text contains no valid tokens. Missing tokens: {missing_tokens}. "
            f"Consider expanding the corpus to include these words."
        )
    
    return valid_tokens

# ============================
# Main Function This is temporary until upgraded models
# ============================

def main():
    """
    Main function to generate vocabulary, validate input, and update the model JSON.
    """
    # Configuration
    json_file = r""  # Path to your model JSON file, remember to use \
    vocab_size = 100000  # Large vocabulary size
    text_corpus = """
        Quantum computing is an area of computing focused on developing computer technology
        based on the principles of quantum theory. Language models aim to generate meaningful
        text based on input, bridging the gap between technology and human interaction.

        Great literature includes works like those of Shakespeare, Dickens, and Austen. 
        Scientific advancements span physics, biology, and chemistry. Everyday conversations 
        include phrases like "hello," "how are you?" and "what's the weather like today?"

        Advanced topics in artificial intelligence, machine learning, and deep learning
        pave the way for innovation. Historical texts, scientific papers, and novels 
        contribute to a rich corpus for modeling.

        Popular culture references, news headlines, and casual dialogue make up modern 
        conversations. The integration of complex fields like robotics and autonomous 
        systems revolutionizes industries. Let's ensure our corpus is broad enough to 
        capture all nuances of human language!
    """  # Expand this with any specific text you wish

    # Generate vocabulary
    print("Generating vocabulary...")
    try:
        token_to_id = generate_vocabulary_from_corpus(text_corpus, vocab_size)
        print(f"Generated {len(token_to_id)} tokens for the vocabulary.")
    except ValueError as e:
        print(f"Error while generating vocabulary: {e}")
        return
    
    # Validate sample input
    sample_input = "hello world, what's quantum computing?"
    print(f"Validating input: '{sample_input}'")
    try:
        valid_tokens = validate_input_text(sample_input, token_to_id)
        print(f"Valid tokens: {valid_tokens}")
    except ValueError as e:
        print(f"Error while validating input: {e}")
        return
    
    # Update the model JSON file
    print("Updating model JSON with vocabulary...")
    try:
        update_model_with_vocabulary(json_file, token_to_id)
        print("Vocabulary update complete.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error while updating model JSON: {e}")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    main()
