#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QELM Conversational UI - Enhanced Basic Version
==========================================

This script provides a chat-style interface to interact with the Quantum-Enhanced Language Model (QELM).
Enhancements include:
1. Support for both .json and .qelm model files.
2. Error handling with output.
3. Modern GUI using ttk.
4. Additional features like clearing chat, saving conversations, and status updates.
5. Added autoloading (if no token file is selected it asks).

Dependencies:
- tkinter (standard with Python)
- numpy
- nltk

Ensure all dependencies are installed before running the script.

Remember that this is a basic chat for the qelm models. This will constantly be updated but not always focused on.

Author: Brenton Carter

==========================================

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import os
import datetime
import traceback  # For detailed error traces

# Initialize NLTK data (only the first time)
nltk.download('punkt', quiet=True)


# ============================
# Quantum Language Model
# ============================

class QuantumLanguageModel:
    """
    Quantum-Enhanced Language Model combining embeddings and output weights.
    Supports loading from both .json and .qelm files along with separate token mapping files.
    """
    def __init__(self):
        self.vocab_size = None
        self.embed_dim = None
        self.hidden_dim = None
        self.embeddings = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.W_out = None  # Output weight matrix
        self.W_proj = None  # Projection matrix (optional)
        self.rotation_angles = None  # If applicable

    def load_from_file(self, model_file_path: str, token_map_file_path: str = None):
        """
        Load model parameters (embeddings and vocab) from a JSON or .qelm file.
        Optionally load token mappings from a separate JSON file.

        Parameters:
            model_file_path (str): Path to the model file.
            token_map_file_path (str, optional): Path to the token mapping JSON file.
        """
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"The file '{model_file_path}' does not exist.")

        _, ext = os.path.splitext(model_file_path)
        if ext.lower() not in ['.json', '.qelm']:
            raise ValueError("Unsupported file format. Please provide a .json or .qelm file.")

        with open(model_file_path, 'r', encoding='utf-8') as f:
            model_dict = json.load(f)

        # Print all keys for verification
        print("Model Keys:")
        for key in model_dict.keys():
            print(f"- {key}")

        # Common required keys
        required_keys = ["vocab_size", "embed_dim", "hidden_dim", "embeddings"]
        for key in required_keys:
            if key not in model_dict:
                raise KeyError(f"Model file is missing the required key: '{key}'")

        self.vocab_size = model_dict["vocab_size"]
        self.embed_dim = model_dict["embed_dim"]
        self.hidden_dim = model_dict["hidden_dim"]
        self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)

        # Load token-to-ID and ID-to-token mappings if present
        if "token_to_id" in model_dict and "id_to_token" in model_dict:
            self.token_to_id = model_dict["token_to_id"]
            self.id_to_token = {int(k): v for k, v in model_dict["id_to_token"].items()}
            print("token_to_id and id_to_token loaded from the model file.")
        else:
            if token_map_file_path:
                self.load_token_map_from_file(token_map_file_path)
            else:
                print("token_to_id and id_to_token not found in the model file and no token mapping file provided.")
                if "vocabulary" in model_dict:
                    tokens = model_dict["vocabulary"]
                    self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                    self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
                    print("Generated 'token_to_id' and 'id_to_token' from model's vocabulary.")
                else:
                    print("Model file does not contain 'vocabulary'. Cannot generate token mappings.")
                    self.token_to_id = {}
                    self.id_to_token = {}

        # Load W_out if present
        if "W_out" in model_dict:
            self.W_out = np.array(model_dict["W_out"], dtype=np.float32)
            expected_shape = (self.vocab_size, self.hidden_dim) if "W_proj" in model_dict and model_dict["W_proj"] is not None else (self.vocab_size, self.embed_dim)
            if self.W_out.shape != expected_shape:
                if "W_proj" in model_dict and model_dict["W_proj"] is not None:
                    # Adjust hidden_dim based on W_out's actual shape
                    new_hidden_dim = self.W_out.shape[1]
                    print(f"Warning: 'W_out' shape mismatch. Adjusting 'hidden_dim' from {self.hidden_dim} to {new_hidden_dim}.")
                    self.hidden_dim = new_hidden_dim
                    # Reinitialize W_proj to match new_hidden_dim and embed_dim
                    self.W_proj = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.01
                    print(f"W_proj reinitialized with shape: {self.W_proj.shape}")
                else:
                    # Adjust embed_dim if W_proj is not present
                    original_embed_dim = self.embed_dim
                    self.embed_dim = self.W_out.shape[1]
                    print(f"Warning: 'W_out' shape mismatch. Adjusting 'embed_dim' from {original_embed_dim} to {self.embed_dim}.")
            print(f"W_out loaded with shape: {self.W_out.shape}")
        else:
            if "W_proj" in model_dict and model_dict["W_proj"] is not None:
                # Initialize W_out with shape (vocab_size, hidden_dim)
                self.W_out = np.random.randn(self.vocab_size, self.hidden_dim).astype(np.float32) * 0.01
                print("Warning: 'W_out' not found in the model file. Initialized randomly with shape (vocab_size, hidden_dim).")
            else:
                # Initialize W_out with shape (vocab_size, embed_dim)
                self.W_out = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.01
                print("Warning: 'W_out' not found in the model file. Initialized randomly with shape (vocab_size, embed_dim).")

        # Load W_proj if present (optional)
        if "W_proj" in model_dict and model_dict["W_proj"] is not None:
            self.W_proj = np.array(model_dict["W_proj"], dtype=np.float32)
            expected_proj_shape = (self.hidden_dim, self.embed_dim)  # Mapping from embed_dim to hidden_dim
            if self.W_proj.shape != expected_proj_shape:
                # Adjust hidden_dim based on W_proj's shape
                new_hidden_dim = self.W_proj.shape[0]
                print(f"Warning: 'W_proj' shape mismatch: expected {expected_proj_shape}, got {self.W_proj.shape}. Adjusting 'hidden_dim' to {new_hidden_dim}.")
                self.hidden_dim = new_hidden_dim
                # Reinitialize W_proj to match new_hidden_dim and embed_dim
                self.W_proj = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.01
                print(f"W_proj reinitialized with shape: {self.W_proj.shape}")
            else:
                print(f"W_proj loaded with shape: {self.W_proj.shape}")
        else:
            self.W_proj = None  # Projection layer not used in UI
            print("W_proj: Not used in this model.")

    def load_token_map_from_file(self, token_map_file_path: str):
        """
        Load token mappings from a separate JSON file.

        Parameters:
            token_map_file_path (str): Path to the token mapping JSON file.
        """
        if not os.path.isfile(token_map_file_path):
            raise FileNotFoundError(f"The token mapping file '{token_map_file_path}' does not exist.")

        try:
            with open(token_map_file_path, 'r', encoding='utf-8') as f:
                token_map = json.load(f)

            if not isinstance(token_map, dict):
                raise ValueError("Token mapping file is not a valid dictionary.")

            # Handle 'token_to_id' and 'id_to_token' presence
            if "token_to_id" in token_map or "id_to_token" in token_map or "vocabulary" in token_map:
                if "token_to_id" in token_map:
                    self.token_to_id = token_map["token_to_id"]
                    if "id_to_token" in token_map:
                        self.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                    else:
                        # Generate 'id_to_token' from 'token_to_id'
                        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
                        print("Generated 'id_to_token' from 'token_to_id'.")
                elif "id_to_token" in token_map:
                    self.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                    # Generate 'token_to_id' from 'id_to_token'
                    self.token_to_id = {v: k for k, v in self.id_to_token.items()}
                    print("Generated 'token_to_id' from 'id_to_token'.")
                elif "vocabulary" in token_map:
                    tokens = token_map["vocabulary"]
                    self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                    self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
                    print("Generated 'token_to_id' and 'id_to_token' from 'vocabulary'.")
            else:
                # Check if the JSON is a flat mapping (tokens as keys and IDs as values)
                if all(isinstance(v, int) for v in token_map.values()):
                    self.token_to_id = token_map
                    self.id_to_token = {v: k for k, v in token_map.items()}
                    print("Generated 'id_to_token' from flat 'token_to_id' mapping.")
                else:
                    raise ValueError("Token mapping file must contain at least 'token_to_id', 'id_to_token', or 'vocabulary'.")

            # Validation Checks
            if len(self.token_to_id) != len(self.id_to_token):
                raise ValueError("Mismatch between 'token_to_id' and 'id_to_token' mappings.")

            # Additional validation: all IDs are unique
            if len(set(self.token_to_id.values())) != len(self.token_to_id.values()):
                raise ValueError("Duplicate IDs found in 'token_to_id' mapping.")

            print(f"Token mapping successfully loaded from '{os.path.basename(token_map_file_path)}'.")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load token mapping from '{token_map_file_path}': {e}")

    def run_inference(self, input_text: str, max_length: int = 10):
        """
        Generate a response based on input text.

        Parameters:
            input_text (str): The user's input message.
            max_length (int): Maximum number of tokens to generate.

        Returns:
            str: The model's response.
        """
        if not self.token_to_id or self.embeddings is None or self.W_out is None:
            raise ValueError("Model is not loaded or embeddings/W_out are missing.")

        # Tokenize and encode input text
        tokens = word_tokenize(input_text.lower())
        input_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]

        if not input_ids:
            available_tokens = ', '.join(list(self.token_to_id.keys())[:10]) + '...'
            raise ValueError("Input text contains no valid tokens. Available tokens include: " + available_tokens)

        # Initialize the response
        response_tokens = []

        # Initialize the input vector
        current_input_ids = input_ids.copy()

        for _ in range(max_length):
            # Use the embeddings to generate a response
            try:
                input_vector = normalize_vector(np.sum(self.embeddings[current_input_ids], axis=0))
            except IndexError as e:
                raise ValueError(f"One of the input_ids {current_input_ids} is out of bounds for embeddings.") from e

            print(f"Input Vector Shape: {input_vector.shape}")  # Debugging

            # If projection layer exists, apply it
            if self.W_proj is not None:
                # W_proj shape: (hidden_dim, embed_dim)
                # input_vector shape: (embed_dim,)
                x = self.W_proj @ input_vector  # Resulting shape: (hidden_dim,)
                print(f"W_proj Shape: {self.W_proj.shape}")  # Debugging
                print(f"Projected Vector Shape: {x.shape}")  # Debugging
            else:
                x = input_vector  # Shape: (embed_dim,)
                print(f"Using Input Vector Directly. Shape: {x.shape}")  # Debugging

            # Compute logits
            print(f"W_out Shape: {self.W_out.shape}")  # Debugging
            print(f"x Shape: {x.shape}")  # Debugging
            logits = self.W_out @ x  # Shape: (vocab_size,)
            print(f"Logits Shape: {logits.shape}")  # Debugging

            # Apply softmax to get probabilities
            probabilities = softmax(logits)

            # Sample a token based on the probability distribution
            try:
                sampled_id = np.random.choice(self.vocab_size, p=probabilities)
            except ValueError as e:
                raise ValueError("Probabilities do not sum to 1 or contain invalid values.") from e

            print(f"Sampled ID: {sampled_id}")  # Debugging

            # Decode the sampled ID back into a token
            sampled_token = self.id_to_token.get(sampled_id, "<UNK>")
            response_tokens.append(sampled_token)

            # If the sampled token is an end token, stop generation
            if sampled_token in [".", "!", "?"]:
                break

            # Update the current input for the next token generation
            current_input_ids = [sampled_id]

        # Combine the tokens into a single string
        response = ' '.join(response_tokens)
        return response


# ============================
# Utility Functions
# ============================

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters:
        vec (np.ndarray): The input vector.

    Returns:
        np.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each set of scores in x.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Softmax probabilities.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_conversation(conversation: list, file_path: str):
    """
    Save the conversation history to a text file.

    Parameters:
        conversation (list): List of conversation lines.
        file_path (str): Path to the file where the conversation will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in conversation:
                f.write(line + '\n')
        messagebox.showinfo("Success", f"Conversation saved to '{file_path}'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save conversation:\n{e}")


# ============================
# Chat UI
# ============================

class QELMChatUI:
    """
    Chat-style User Interface for interacting with the QELM.
    Enhanced with better styling, error handling, and additional features.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("QELM Chat")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Initialize model
        self.model = QuantumLanguageModel()

        # Initialize conversation history
        self.conversation = []

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use 'clam' for better aesthetics

        # Define color scheme
        self.bg_color = "#2E3440"  # Dark background
        self.text_color = "#D8DEE9"  # Light text
        self.user_color = "#81A1C1"  # User messages color
        self.qelm_color = "#A3BE8C"  # QELM responses color
        self.system_color = "#BF616A"  # System messages color

        # Set window background
        self.root.configure(bg=self.bg_color)

        # Create frames
        self.create_frames()

        # Create widgets
        self.create_widgets()

        # Bind Enter key to send message
        self.user_input.bind("<Return>", self.handle_send)

    def create_frames(self):
        """
        Create and organize frames within the main window.
        """
        self.chat_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.chat_frame.pack(fill="both", expand=True)

        self.input_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.input_frame.pack(fill="x")

        self.button_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.button_frame.pack(fill="x")

        self.status_frame = ttk.Frame(self.root, padding="5 5 5 5")
        self.status_frame.pack(fill="x")

    def create_widgets(self):
        """
        Create and place widgets within the frames.
        """
        # Chat display with scrollbar
        self.chat_display = tk.Text(
            self.chat_frame,
            height=25,
            bg=self.bg_color,
            fg=self.text_color,
            font=("Helvetica", 12),
            wrap="word",
            state="disabled",
            relief="flat",
            highlightthickness=0
        )
        self.chat_display.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_display.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_display['yscrollcommand'] = self.scrollbar.set

        # User input field
        self.user_input = ttk.Entry(self.input_frame, font=("Helvetica", 12))
        self.user_input.pack(fill="x", expand=True, side="left", padx=(0, 10))
        self.user_input.focus()

        # Send button
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.handle_send)
        self.send_button.pack(side="right")

        # Load Model button
        self.load_button = ttk.Button(self.button_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side="left", padx=(0, 10))

        # Clear Chat button
        self.clear_button = ttk.Button(self.button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side="left", padx=(0, 10))

        # Save Conversation button
        self.save_button = ttk.Button(self.button_frame, text="Save Conversation", command=self.save_chat)
        self.save_button.pack(side="left", padx=(0, 10))

        # Status bar
        self.status_label = ttk.Label(
            self.status_frame,
            text="No model loaded.",
            anchor="w",
            font=("Helvetica", 10)
        )
        self.status_label.pack(fill="x")

    def load_model(self):
        """
        Open a file dialog to load the model JSON or .qelm file.
        If 'token_to_id' is missing, prompt the user to load the token mapping JSON file.
        """
        # Prompt user to select the model file
        model_file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.json *.qelm"), ("All Files", "*.*")]
        )
        if not model_file_path:
            return

        try:
            # Attempt to load the model file without token mapping
            self.model.load_from_file(model_file_path)
            print("Model loaded without token mapping.")
            self.update_chat("System", f"Model loaded from '{os.path.basename(model_file_path)}'.", color=self.system_color)
            self.status_label.config(text=f"Model loaded: {os.path.basename(model_file_path)}")

            if not self.model.token_to_id or not self.model.id_to_token:
                # Prompt user to load token mapping
                response = messagebox.askyesno(
                    "Token Mapping Missing",
                    "The model file does not contain 'token_to_id', 'id_to_token', or 'vocabulary' mappings. Would you like to load it from a separate JSON file?"
                )
                if response:
                    self.prompt_token_map_loading()
                else:
                    messagebox.showwarning("Token Mapping Not Loaded", "Inference may not work correctly without token mapping.")
                    self.status_label.config(text="Model loaded without token mapping.")
            else:
                self.status_label.config(text=f"Model loaded with token mapping.")
                # List available tokens for verification
                self.display_available_tokens()

        except Exception as e:
            error_message = f"Failed to load model:\n{e}\n{traceback.format_exc()}"
            self.update_chat("System", error_message, color=self.system_color)
            messagebox.showerror("Load Error", error_message)
            self.status_label.config(text="Failed to load model.")

    def prompt_token_map_loading(self):
        """
        Prompt the user to select the token mapping JSON file.
        """
        token_map_path = filedialog.askopenfilename(
            title="Select Token Mapping JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not token_map_path:
            messagebox.showwarning("Token Mapping Not Loaded", "Inference may not work correctly without token mapping.")
            self.status_label.config(text="Model loaded without token mapping.")
            return

        try:
            self.model.load_token_map_from_file(token_map_path)
            self.update_chat("System", f"Token mapping loaded from '{os.path.basename(token_map_path)}'.", color=self.system_color)
            self.status_label.config(text=f"Token mapping loaded: {os.path.basename(token_map_path)}")
            # List available tokens for verification
            self.display_available_tokens()
        except Exception as e:
            error_message = f"Failed to load token mapping:\n{e}\n{traceback.format_exc()}"
            self.update_chat("System", error_message, color=self.system_color)
            messagebox.showerror("Token Mapping Load Error", error_message)
            self.status_label.config(text="Failed to load token mapping.")

    def handle_send(self, event=None):
        """
        Process user input and display the model's response.
        Triggered by Send button or Enter key.
        """
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        # Display user message
        self.update_chat("User", user_text, color=self.user_color)

        response = ""  # Initialize response

        try:
            # Generate and display model response
            response = self.model.run_inference(user_text)
            self.update_chat("QELM", response, color=self.qelm_color)
            self.status_label.config(text="Response generated.")
        except Exception as e:
            error_message = f"Error: {e}"
            self.update_chat("System", error_message, color=self.system_color)
            self.status_label.config(text="Error during inference.")
            response = "<Error: Response generation failed>"

        # Add to conversation history
        self.conversation.append(f"User: {user_text}")
        self.conversation.append(f"QELM: {response}")

        # Clear input field
        self.user_input.delete(0, tk.END)

    def update_chat(self, sender: str, message: str, color: str = None):
        """
        Update the chat display with a new message.

        Parameters:
            sender (str): The sender of the message (e.g., User, QELM, System).
            message (str): The message content.
            color (str, optional): The color for the message text.
        """
        self.chat_display.config(state="normal")
        if sender == "User":
            tag = "user"
        elif sender == "QELM":
            tag = "qelm"
        else:
            tag = "system"

        # Configure tag if not already
        if tag not in self.chat_display.tag_names():
            self.chat_display.tag_configure(tag, foreground=color if color else self.text_color, font=("Helvetica", 12, "bold"))

        # Insert message
        self.chat_display.insert(tk.END, f"{sender}: {message}\n", tag)
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def clear_chat(self):
        """
        Clear the chat display and conversation history.
        """
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to clear the chat?")
        if confirm:
            self.chat_display.config(state="normal")
            self.chat_display.delete('1.0', tk.END)
            self.chat_display.config(state="disabled")
            self.conversation.clear()
            self.status_label.config(text="Chat cleared.")

    def save_chat(self):
        """
        Save the conversation history to a text file.
        """
        if not self.conversation:
            messagebox.showinfo("Info", "No conversation to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Conversation"
        )
        if not file_path:
            return

        try:
            # Include timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Conversation saved on {timestamp}\n\n")
                for line in self.conversation:
                    f.write(line + '\n')
            messagebox.showinfo("Success", f"Conversation saved to '{file_path}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation:\n{e}")

    def display_available_tokens(self):
        """
        Display a list of available tokens in the model.
        Useful for debugging purposes.
        """
        if not self.model.token_to_id:
            self.update_chat("System", "No token mappings available.", color=self.system_color)
            return

        tokens = list(self.model.token_to_id.keys())
        tokens_display = ', '.join(tokens[:50]) + ('...' if len(tokens) > 50 else '')
        self.update_chat("System", f"Available tokens: {tokens_display}", color=self.system_color)


# ============================
# Main Entry Point
# ============================

def main():
    """
    Initialize and run the QELM Chat UI.
    """
    try:
        root = tk.Tk()
        app = QELMChatUI(root)
        root.mainloop()
    except Exception as e:
        error_trace = traceback.format_exc()
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n{error_trace}")
        print(f"Fatal Error: {e}\n{error_trace}")


if __name__ == "__main__":
    main()
