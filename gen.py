#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to Generate Vocabulary from a Large Corpus or Dataset and Update a Model Qelm/JSON with a Tkinter GUI.
You can alter this in any way as it's very rudi and may not reflect current models as the base may change consistantly. 
"""

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
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


class VocabularyGUI:
    def __init__(self, master):
        self.master = master
        master.title("Vocabulary Generator and Model Updater")

        # Initialize variables
        self.json_file = tk.StringVar()
        self.vocab_size = tk.IntVar(value=100000)
        self.sample_input = tk.StringVar()
        self.corpus_mode = tk.StringVar(value="manual")
        self.dataset_path = tk.StringVar()

        # Layout configuration
        self.create_widgets()

    def create_widgets(self):
        # Mode Selection
        mode_frame = tk.Frame(self.master)
        mode_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(mode_frame, text="Corpus Source:").pack(anchor=tk.W)
        modes = [("Manual Text Corpus", "manual"), ("Use Dataset", "dataset")]
        for text, mode in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.corpus_mode, value=mode, command=self.toggle_corpus_source).pack(anchor=tk.W)

        # Dataset Selection
        self.dataset_frame = tk.Frame(self.master)
        self.dataset_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(self.dataset_frame, text="Dataset File/Directory:").pack(side=tk.LEFT)
        self.dataset_entry = tk.Entry(self.dataset_frame, textvariable=self.dataset_path, width=50, state='disabled')
        self.dataset_entry.pack(side=tk.LEFT, padx=5)
        self.browse_dataset_button = tk.Button(self.dataset_frame, text="Browse", command=self.browse_dataset, state='disabled')
        self.browse_dataset_button.pack(side=tk.LEFT)

        # JSON File Selection
        json_frame = tk.Frame(self.master)
        json_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(json_frame, text="Model JSON File:").pack(side=tk.LEFT)
        self.json_entry = tk.Entry(json_frame, textvariable=self.json_file, width=50)
        self.json_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(json_frame, text="Browse", command=self.browse_json).pack(side=tk.LEFT)

        # Vocabulary Size
        vocab_frame = tk.Frame(self.master)
        vocab_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(vocab_frame, text="Vocabulary Size:").pack(side=tk.LEFT)
        self.vocab_entry = tk.Entry(vocab_frame, textvariable=self.vocab_size, width=10)
        self.vocab_entry.pack(side=tk.LEFT, padx=5)

        # Text Corpus
        self.corpus_frame = tk.Frame(self.master)
        self.corpus_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(self.corpus_frame, text="Text Corpus:").pack(anchor=tk.W)
        self.corpus_text = scrolledtext.ScrolledText(self.corpus_frame, height=10)
        self.corpus_text.pack(fill=tk.BOTH, expand=True)
        # Insert default corpus
        default_corpus = """
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
"""
        self.corpus_text.insert(tk.END, default_corpus.strip())

        # Sample Input
        sample_frame = tk.Frame(self.master)
        sample_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(sample_frame, text="Sample Input:").pack(side=tk.LEFT)
        self.sample_entry = tk.Entry(sample_frame, textvariable=self.sample_input, width=50)
        self.sample_entry.pack(side=tk.LEFT, padx=5)
        self.sample_entry.insert(0, "hello world, what's quantum computing?")

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(padx=10, pady=10)

        tk.Button(button_frame, text="Generate Vocabulary", command=self.generate_vocabulary).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Validate Input", command=self.validate_input).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Update Model JSON", command=self.update_model).pack(side=tk.LEFT, padx=5)

        # Log Area
        log_frame = tk.Frame(self.master)
        log_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(log_frame, text="Log:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def toggle_corpus_source(self):
        mode = self.corpus_mode.get()
        if mode == "manual":
            self.dataset_entry.config(state='disabled')
            self.browse_dataset_button.config(state='disabled')
            self.corpus_text.config(state='normal')
            self.log("Switched to Manual Text Corpus mode.")
        else:
            self.dataset_entry.config(state='normal')
            self.browse_dataset_button.config(state='normal')
            self.corpus_text.config(state='disabled')
            self.log("Switched to Dataset mode.")

    def browse_json(self):
        filepath = filedialog.askopenfilename(
            title="Select Model JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self.json_file.set(filepath)
            self.log(f"Selected JSON file: {filepath}")

    def browse_dataset(self):
        filepath = filedialog.askopenfilename(
            title="Select Dataset File or Directory",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            # If no file is selected, try selecting a directory
            filepath = filedialog.askdirectory(title="Select Dataset Directory")
        if filepath:
            self.dataset_path.set(filepath)
            self.log(f"Selected dataset path: {filepath}")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def load_corpus_from_dataset(self, path):
        """
        Load text from a single file or all text files in a directory.
        """
        if os.path.isfile(path):
            if not path.lower().endswith('.txt'):
                raise ValueError("Selected file is not a text file (*.txt).")
            with open(path, 'r', encoding='utf-8') as f:
                corpus = f.read()
            self.log(f"Loaded corpus from file: {path}")
            return corpus
        elif os.path.isdir(path):
            corpus = ""
            txt_files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
            if not txt_files:
                raise ValueError("No text files (*.txt) found in the selected directory.")
            for file in txt_files:
                file_path = os.path.join(path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    corpus += f.read() + " "
                self.log(f"Loaded text from: {file_path}")
            return corpus
        else:
            raise ValueError("Selected path is neither a file nor a directory.")

    def generate_vocabulary(self):
        self.log("Generating vocabulary...")
        mode = self.corpus_mode.get()
        vocab_size = self.vocab_size.get()

        try:
            if mode == "manual":
                corpus = self.corpus_text.get("1.0", tk.END)
                token_to_id = generate_vocabulary_from_corpus(corpus, vocab_size)
                self.log(f"Generated {len(token_to_id)} tokens for the vocabulary from manual corpus.")
            else:
                dataset_path = self.dataset_path.get()
                if not dataset_path:
                    raise ValueError("Dataset path is not specified.")
                corpus = self.load_corpus_from_dataset(dataset_path)
                token_to_id = generate_vocabulary_from_corpus(corpus, vocab_size)
                self.log(f"Generated {len(token_to_id)} tokens for the vocabulary from dataset.")
            
            messagebox.showinfo("Success", f"Vocabulary generated with {len(token_to_id)} tokens.")
            self.generated_token_to_id = token_to_id  # Store for later use
        except ValueError as e:
            self.log(f"Error while generating vocabulary: {e}")
            messagebox.showerror("Error", f"Failed to generate vocabulary:\n{e}")

    def validate_input(self):
        self.log("Validating sample input...")
        sample = self.sample_input.get()
        try:
            if not hasattr(self, 'generated_token_to_id'):
                raise ValueError("Vocabulary not generated yet. Please generate vocabulary first.")
            valid_tokens = validate_input_text(sample, self.generated_token_to_id)
            self.log(f"Valid tokens: {valid_tokens}")
            messagebox.showinfo("Validation Success", f"Valid tokens:\n{valid_tokens}")
        except ValueError as e:
            self.log(f"Error while validating input: {e}")
            messagebox.showerror("Validation Error", f"Failed to validate input:\n{e}")

    def update_model(self):
        self.log("Updating model JSON with vocabulary...")
        json_file = self.json_file.get()
        try:
            if not hasattr(self, 'generated_token_to_id'):
                raise ValueError("Vocabulary not generated yet. Please generate vocabulary first.")
            if not json_file:
                raise ValueError("JSON file path is not specified.")
            update_model_with_vocabulary(json_file, self.generated_token_to_id)
            self.log("Vocabulary update complete.")
            messagebox.showinfo("Success", "Model JSON updated successfully.")
        except (FileNotFoundError, ValueError) as e:
            self.log(f"Error while updating model JSON: {e}")
            messagebox.showerror("Update Error", f"Failed to update model JSON:\n{e}")


# ============================
# Entry Point
# ============================

def main():
    root = tk.Tk()
    gui = VocabularyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
