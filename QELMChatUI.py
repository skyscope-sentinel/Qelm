#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QELM Conversational UI - Rudi judi
============================================================================
This script provides a chat-style interface to interact with the Quantum-Enhanced
Language Model (QELM), with an enhanced layout similar to modern chat interfaces
(e.g., ChatGPT). The duplication issue in user and QELM messages is resolved by
avoiding multiple appends to the conversation history.

Author: Brenton Carter (modified to fix duplication)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import os
import datetime
import traceback  

# Initialize NLTK data (only the first time)
nltk.download('punkt', quiet=True)


def normalize_vector(vec: np.ndarray) -> np.ndarray:

    # Normalize a vector to unit length.

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


def softmax(x: np.ndarray) -> np.ndarray:

    # Compute softmax values for each set of scores in x.

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_conversation(conversation: list, file_path: str):

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in conversation:
                f.write(line + '\n')
        messagebox.showinfo("Success", f"Conversation saved to '{file_path}'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save conversation:\n{e}")


class QuantumLanguageModel:

    def __init__(self):
        self.vocab_size = None
        self.embed_dim = None
        self.hidden_dim = None
        self.embeddings = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.W_out = None
        self.W_proj = None
        self.rotation_angles = None

    def load_from_file(self, model_file_path: str, token_map_file_path: str = None):
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"The file '{model_file_path}' does not exist.")

        _, ext = os.path.splitext(model_file_path)
        if ext.lower() not in ['.json', '.qelm']:
            raise ValueError("Unsupported file format. Please provide a .json or .qelm file.")

        with open(model_file_path, 'r', encoding='utf-8') as f:
            model_dict = json.load(f)

        print("Model Keys:")
        for key in model_dict.keys():
            print(f"- {key}")

        required_keys = ["vocab_size", "embed_dim", "hidden_dim", "embeddings"]
        for key in required_keys:
            if key not in model_dict:
                raise KeyError(f"Model file is missing the required key: '{key}'")

        self.vocab_size = model_dict["vocab_size"]
        self.embed_dim = model_dict["embed_dim"]
        self.hidden_dim = model_dict["hidden_dim"]
        self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)

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

        if "W_out" in model_dict:
            self.W_out = np.array(model_dict["W_out"], dtype=np.float32)
            expected_shape = (self.vocab_size, self.hidden_dim) if "W_proj" in model_dict and model_dict["W_proj"] is not None else (self.vocab_size, self.embed_dim)
            if self.W_out.shape != expected_shape:
                if "W_proj" in model_dict and model_dict["W_proj"] is not None:
                    new_hidden_dim = self.W_out.shape[1]
                    print(f"Warning: 'W_out' shape mismatch. Adjusting 'hidden_dim' from {self.hidden_dim} to {new_hidden_dim}.")
                    self.hidden_dim = new_hidden_dim
                    self.W_proj = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.01
                    print(f"W_proj reinitialized with shape: {self.W_proj.shape}")
                else:
                    original_embed_dim = self.embed_dim
                    self.embed_dim = self.W_out.shape[1]
                    print(f"Warning: 'W_out' shape mismatch. Adjusting 'embed_dim' from {original_embed_dim} to {self.embed_dim}.")
            print(f"W_out loaded with shape: {self.W_out.shape}")
        else:
            if "W_proj" in model_dict and model_dict["W_proj"] is not None:
                self.W_out = np.random.randn(self.vocab_size, self.hidden_dim).astype(np.float32) * 0.01
                print("Warning: 'W_out' not found. Initialized randomly with shape (vocab_size, hidden_dim).")
            else:
                self.W_out = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.01
                print("Warning: 'W_out' not found. Initialized randomly with shape (vocab_size, embed_dim).")

        if "W_proj" in model_dict and model_dict["W_proj"] is not None:
            self.W_proj = np.array(model_dict["W_proj"], dtype=np.float32)
            expected_proj_shape = (self.hidden_dim, self.embed_dim)
            if self.W_proj.shape != expected_proj_shape:
                new_hidden_dim = self.W_proj.shape[0]
                print(f"Warning: 'W_proj' shape mismatch: expected {expected_proj_shape}, got {self.W_proj.shape}. Adjusting 'hidden_dim' to {new_hidden_dim}.")
                self.hidden_dim = new_hidden_dim
                self.W_proj = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.01
                print(f"W_proj reinitialized with shape: {self.W_proj.shape}")
            else:
                print(f"W_proj loaded with shape: {self.W_proj.shape}")
        else:
            self.W_proj = None
            print("W_proj: Not used in this model.")

    def load_token_map_from_file(self, token_map_file_path: str):
        if not os.path.isfile(token_map_file_path):
            raise FileNotFoundError(f"The token mapping file '{token_map_file_path}' does not exist.")

        try:
            with open(token_map_file_path, 'r', encoding='utf-8') as f:
                token_map = json.load(f)

            if not isinstance(token_map, dict):
                raise ValueError("Token mapping file is not a valid dictionary.")

            if "token_to_id" in token_map or "id_to_token" in token_map or "vocabulary" in token_map:
                if "token_to_id" in token_map:
                    self.token_to_id = token_map["token_to_id"]
                    if "id_to_token" in token_map:
                        self.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                    else:
                        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
                        print("Generated 'id_to_token' from 'token_to_id'.")
                elif "id_to_token" in token_map:
                    self.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                    self.token_to_id = {v: k for k, v in self.id_to_token.items()}
                    print("Generated 'token_to_id' from 'id_to_token'.")
                elif "vocabulary" in token_map:
                    tokens = token_map["vocabulary"]
                    self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                    self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
                    print("Generated 'token_to_id' and 'id_to_token' from 'vocabulary'.")
            else:
                if all(isinstance(v, int) for v in token_map.values()):
                    self.token_to_id = token_map
                    self.id_to_token = {v: k for k, v in token_map.items()}
                    print("Generated 'id_to_token' from flat 'token_to_id' mapping.")
                else:
                    raise ValueError("Token map must have 'token_to_id', 'id_to_token', or 'vocabulary'.")

            if len(self.token_to_id) != len(self.id_to_token):
                raise ValueError("Mismatch between 'token_to_id' and 'id_to_token' mappings.")
            if len(set(self.token_to_id.values())) != len(self.token_to_id.values()):
                raise ValueError("Duplicate IDs found in 'token_to_id' mapping.")

            print(f"Token mapping successfully loaded from '{os.path.basename(token_map_file_path)}'.")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load token mapping from '{token_map_file_path}': {e}")

    def run_inference(self, input_text: str, max_length: int = 10):
        if not self.token_to_id or self.embeddings is None or self.W_out is None:
            raise ValueError("Model is not loaded or embeddings/W_out are missing.")

        tokens = word_tokenize(input_text.lower())
        input_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]

        if not input_ids:
            available_tokens = ', '.join(list(self.token_to_id.keys())[:10]) + '...'
            raise ValueError("Input text contains no valid tokens. Available tokens include: " + available_tokens)

        response_tokens = []
        current_input_ids = input_ids.copy()

        for _ in range(max_length):
            try:
                input_vector = normalize_vector(np.sum(self.embeddings[current_input_ids], axis=0))
            except IndexError as e:
                raise ValueError(f"One of the input_ids {current_input_ids} is out of bounds.") from e

            if self.W_proj is not None:
                x = self.W_proj @ input_vector
            else:
                x = input_vector

            logits = self.W_out @ x
            probabilities = softmax(logits)

            try:
                sampled_id = np.random.choice(self.vocab_size, p=probabilities)
            except ValueError as e:
                raise ValueError("Probabilities do not sum to 1 or contain invalid values.") from e

            sampled_token = self.id_to_token.get(sampled_id, "<UNK>")
            response_tokens.append(sampled_token)

            if sampled_token in [".", "!", "?"]:
                break

            current_input_ids = [sampled_id]

        response = ' '.join(response_tokens)
        return response


class QELMChatUI:
# Below are modifiable values for the UI.
    def __init__(self, root):
        self.root = root
        self.root.title("QELM Chat - Gpt layout")
        self.root.geometry("1100x600")
        self.root.resizable(False, False)

        # Initialize model
        self.model = QuantumLanguageModel()

        # Dictionary of conversations, keyed by conversation name/ID
        self.conversations = {}
        self.current_conversation_id = None

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.bg_color = "#2E3440"
        self.text_color = "#D8DEE9"
        self.user_color = "#81A1C1"
        self.qelm_color = "#A3BE8C"
        self.system_color = "#BF616A"

        self.root.configure(bg=self.bg_color)

        self.create_main_layout()
        self.create_side_panel()
        self.create_top_panel()
        self.create_chat_panel()
        self.create_bottom_panel()

        # Create a default conversation
        self.new_conversation("Default")

        # Bind Enter key
        self.user_input.bind("<Return>", self.handle_send)

    def create_main_layout(self):
        self.left_frame = ttk.Frame(self.root, width=250)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.top_panel_frame = ttk.Frame(self.right_frame)
        self.top_panel_frame.pack(side="top", fill="x")

        self.main_panel_frame = ttk.Frame(self.right_frame)
        self.main_panel_frame.pack(side="top", fill="both", expand=True)

        self.bottom_panel_frame = ttk.Frame(self.right_frame)
        self.bottom_panel_frame.pack(side="bottom", fill="x")

    def create_side_panel(self):
        side_label = ttk.Label(self.left_frame, text="Conversations", font=("Helvetica", 14, "bold"))
        side_label.pack(side="top", pady=5)

        self.convo_listbox = tk.Listbox(
            self.left_frame, bg=self.bg_color, fg=self.text_color, font=("Helvetica", 12),
            selectbackground="#434C5E", selectforeground="#ECEFF4", height=20
        )
        self.convo_listbox.pack(side="top", fill="both", expand=True)
        self.convo_listbox.bind("<<ListboxSelect>>", self.handle_conversation_select)

        new_convo_button = ttk.Button(self.left_frame, text="New Conversation", command=self.handle_new_conversation)
        new_convo_button.pack(side="bottom", pady=10)

    def create_top_panel(self):
        ttk.Label(self.top_panel_frame, text="Model:", font=("Helvetica", 12, "bold")).pack(side="left", padx=(0, 5))

        self.model_combo = ttk.Combobox(self.top_panel_frame, values=[], width=50)
        self.model_combo.pack(side="left", padx=(0, 5))
        self.model_combo.set("Select or Load Model...")

        load_button = ttk.Button(self.top_panel_frame, text="Load Model", command=self.load_model)
        load_button.pack(side="left", padx=(5, 5))

        self.status_label = ttk.Label(
            self.top_panel_frame,
            text="No model loaded",
            font=("Helvetica", 10)
        )
        self.status_label.pack(side="right", padx=5)

    def create_chat_panel(self):
        self.chat_display = tk.Text(
            self.main_panel_frame,
            bg=self.bg_color,
            fg=self.text_color,
            font=("Helvetica", 12),
            wrap="word",
            state="disabled",
            relief="flat",
            highlightthickness=0
        )
        self.chat_display.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(self.main_panel_frame, orient="vertical", command=self.chat_display.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_display['yscrollcommand'] = self.scrollbar.set

    def create_bottom_panel(self):
        self.user_input = ttk.Entry(self.bottom_panel_frame, font=("Helvetica", 12))
        self.user_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.user_input.focus()

        self.send_button = ttk.Button(self.bottom_panel_frame, text="Send", command=self.handle_send)
        self.send_button.pack(side="left")

        self.save_button = ttk.Button(self.bottom_panel_frame, text="Save Chat", command=self.save_chat)
        self.save_button.pack(side="left", padx=(5, 5))

        self.clear_button = ttk.Button(self.bottom_panel_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side="left")

    def new_conversation(self, title: str):
        if title in self.conversations:
            suffix = 1
            base_title = title
            while title in self.conversations:
                suffix += 1
                title = f"{base_title} ({suffix})"

        self.conversations[title] = []
        self.convo_listbox.insert(tk.END, title)
        self.convo_listbox.selection_clear(0, tk.END)
        last_index = self.convo_listbox.size() - 1
        self.convo_listbox.selection_set(last_index)
        self.convo_listbox.event_generate("<<ListboxSelect>>")

    def handle_new_conversation(self):
        conv_name = f"Conversation {len(self.conversations) + 1}"
        self.new_conversation(conv_name)

    def handle_conversation_select(self, event=None):
        selection = self.convo_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        convo_title = self.convo_listbox.get(index)
        self.switch_conversation(convo_title)

    def switch_conversation(self, convo_title: str):
        self.current_conversation_id = convo_title
        self.refresh_chat_display()

    def get_current_conversation(self) -> list:
        if self.current_conversation_id is None:
            return []
        return self.conversations[self.current_conversation_id]

    def refresh_chat_display(self):
        self.chat_display.config(state="normal")
        self.chat_display.delete('1.0', tk.END)
        conversation_data = self.get_current_conversation()

        for line in conversation_data:
            if line.startswith("User:"):
                self.chat_display.insert(tk.END, line + "\n", "user")
            elif line.startswith("QELM:"):
                self.chat_display.insert(tk.END, line + "\n", "qelm")
            else:
                self.chat_display.insert(tk.END, line + "\n", "system")

        self.chat_display.tag_configure("user", foreground=self.user_color, font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("qelm", foreground=self.qelm_color, font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("system", foreground=self.system_color, font=("Helvetica", 12, "italic"))

        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def load_model(self):
        model_file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.json *.qelm"), ("All Files", "*.*")]
        )
        if not model_file_path:
            return

        try:
            self.model.load_from_file(model_file_path)
            self.add_model_to_combo(model_file_path)
            msg = f"Model loaded from '{os.path.basename(model_file_path)}'."
            self.update_chat("System", msg, color=self.system_color)
            self.status_label.config(text=msg)

            if not self.model.token_to_id or not self.model.id_to_token:
                response = messagebox.askyesno(
                    "Token Mapping Missing",
                    "The model file does not contain 'token_to_id', 'id_to_token', or 'vocabulary'. Load a separate JSON?"
                )
                if response:
                    self.prompt_token_map_loading()
                else:
                    messagebox.showwarning("Token Mapping Not Loaded", "Inference may not work correctly.")
                    self.status_label.config(text="Model loaded without token mapping.")
            else:
                self.status_label.config(text="Model loaded with token mapping.")
                self.display_available_tokens()

        except Exception as e:
            error_message = f"Failed to load model:\n{e}\n{traceback.format_exc()}"
            self.update_chat("System", error_message, color=self.system_color)
            messagebox.showerror("Load Error", error_message)
            self.status_label.config(text="Failed to load model.")

    def add_model_to_combo(self, model_path: str):
        current_values = list(self.model_combo["values"])
        if model_path not in current_values:
            current_values.append(model_path)
            self.model_combo["values"] = current_values
        self.model_combo.set(model_path)

    def prompt_token_map_loading(self):
        token_map_path = filedialog.askopenfilename(
            title="Select Token Mapping JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not token_map_path:
            messagebox.showwarning("Token Mapping Not Loaded", "Inference may not work correctly.")
            self.status_label.config(text="Model loaded without token mapping.")
            return

        try:
            self.model.load_token_map_from_file(token_map_path)
            msg = f"Token mapping loaded from '{os.path.basename(token_map_path)}'."
            self.update_chat("System", msg, color=self.system_color)
            self.status_label.config(text=msg)
            self.display_available_tokens()
        except Exception as e:
            error_message = f"Failed to load token mapping:\n{e}\n{traceback.format_exc()}"
            self.update_chat("System", error_message, color=self.system_color)
            messagebox.showerror("Token Mapping Load Error", error_message)
            self.status_label.config(text="Failed to load token mapping.")

    # Tokenization values (may need to be altered for embedding issues) 
    
    def display_available_tokens(self):
        if not self.model.token_to_id:
            self.update_chat("System", "No token mappings available.", color=self.system_color)
            return
        tokens = list(self.model.token_to_id.keys())
        tokens_display = ', '.join(tokens[:50]) + ('...' if len(tokens) > 50 else '')
        self.update_chat("System", f"Available tokens: {tokens_display}", color=self.system_color)

    def handle_send(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.update_chat("User", user_text, color=self.user_color)

        response = ""
        try:
            response = self.model.run_inference(user_text)
            self.update_chat("QELM", response, color=self.qelm_color)
            self.status_label.config(text="Response generated.")
        except Exception as e:
            error_message = f"Error: {e}"
            self.update_chat("System", error_message, color=self.system_color)
            self.status_label.config(text="Error during inference.")
            response = "<Error: Response generation failed>"

        self.refresh_chat_display()
        self.user_input.delete(0, tk.END)

    def update_chat(self, sender: str, message: str, color: str = None):
        if self.current_conversation_id:
            self.conversations[self.current_conversation_id].append(f"{sender}: {message}")
            self.refresh_chat_display()
        else:
            if len(self.conversations) == 0:
                self.new_conversation("Default")
            self.conversations["Default"].append(f"{sender}: {message}")
            self.refresh_chat_display()

    def clear_chat(self):
        if not self.current_conversation_id:
            return
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to clear this conversation?")
        if confirm:
            self.conversations[self.current_conversation_id].clear()
            self.refresh_chat_display()
            self.status_label.config(text="Chat cleared.")

    def save_chat(self):
        if not self.current_conversation_id:
            messagebox.showinfo("Info", "No conversation selected to save.")
            return

        conversation = self.conversations[self.current_conversation_id]
        if not conversation:
            messagebox.showinfo("Info", "No messages to save in this conversation.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Conversation"
        )
        if not file_path:
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Conversation: {self.current_conversation_id}\n")
                f.write(f"Saved on: {timestamp}\n\n")
                for line in conversation:
                    f.write(line + "\n")
            messagebox.showinfo("Success", f"Conversation saved to '{file_path}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation:\n{e}")

def main():
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
