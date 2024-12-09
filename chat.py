#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QELM Conversational UI
This script provides a chat-style interface to interact with the Quantum-Enhanced Language Model (QELM). Super dooper rudi.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

# ============================
# Quantum Language Model
# ============================

class QuantumLanguageModel:
    def __init__(self):
        self.vocab_size = None
        self.embed_dim = None
        self.embeddings = None
        self.token_to_id = None
        self.id_to_token = None

    def load_from_file(self, file_path: str):
        """
        Load model parameters (embeddings and vocab) from a JSON file.
        """
        with open(file_path, 'r') as f:
            model_dict = json.load(f)

        self.vocab_size = model_dict["vocab_size"]
        self.embed_dim = model_dict["embed_dim"]
        self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)

        # Load token-to-ID and ID-to-token mappings if available
        if "token_to_id" in model_dict:
            self.token_to_id = model_dict["token_to_id"]
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        else:
            raise ValueError("Model JSON does not contain token mappings.")

    def run_inference(self, input_text: str):
        """
        Generate a response based on input text.
        """
        if not self.token_to_id or not self.embeddings.size:
            raise ValueError("Model is not loaded or embeddings are missing.")

        # Tokenize and encode input text
        tokens = word_tokenize(input_text.lower())
        input_ids = [self.token_to_id[token] for token in tokens if token in self.token_to_id]

        if not input_ids:
            raise ValueError("Input text contains no valid tokens.")

        # Use the embeddings to generate a response
        input_vector = normalize_vector(np.sum(self.embeddings[input_ids], axis=0))
        logits = np.dot(self.embeddings, input_vector)
        top_response_id = np.argmax(logits)

        # Decode the top response ID back into a token
        response_token = self.id_to_token.get(top_response_id, "<UNK>")
        return response_token

# ============================
# Utility Functions
# ============================

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec

# ============================
# Chat UI
# ============================

class QELMChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QELM Chat")
        self.root.geometry("600x500")
        self.model = QuantumLanguageModel()

        # Chat display
        self.chat_display = tk.Text(root, height=20, state="disabled", font=("Arial", 12), wrap="word")
        self.chat_display.pack(padx=10, pady=10, fill="both", expand=True)

        # User input
        self.user_input = tk.Entry(root, font=("Arial", 12))
        self.user_input.pack(padx=10, pady=5, fill="x", expand=True)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.handle_send, font=("Arial", 12))
        self.send_button.pack(pady=5)

        # Load Model button
        self.load_button = tk.Button(root, text="Load Model", command=self.load_model, font=("Arial", 12))
        self.load_button.pack(pady=5)

    def load_model(self):
        """
        Open a file dialog to load the model JSON file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.model.load_from_file(file_path)
            self.update_chat("System", f"Model loaded successfully from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def handle_send(self):
        """
        Process user input and display the model's response.
        """
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        # Display user message
        self.update_chat("User", user_text)

        try:
            # Generate and display model response
            response = self.model.run_inference(user_text)
            self.update_chat("QELM", response)
        except Exception as e:
            self.update_chat("System", f"Error: {e}")

        # Clear input field
        self.user_input.delete(0, tk.END)

    def update_chat(self, sender, message):
        """
        Update the chat display with a new message.
        """
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

# ============================
# Main Entry Point
# ============================

if __name__ == "__main__":
    root = tk.Tk()
    app = QELMChatUI(root)
    root.mainloop()
