#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================================================
Quantum-Enhanced Language Model (QELM) - Neuron
====================================================================================================

This script defines an optimized Quantum-Enhanced Language Model (QELM) leveraging Qiskit while
utilizing qubit architecture while mapping out quantum states for creating QELM's. 
It incorporates advanced quantum techniques such as logical qubits with error correction,
graphene-like layer structures, and scalable neural network architectures inspired by string theory
and brain-like connectivity. The model is designed to simulate a quantum neural network
aiming to achieve efficient and fault-tolerant language processing capabilities.

Key Components:
1. Quantum Neuron (20-Qubit Logical Unit)
2. Graphene-Like Layer Structure
3. Stacked Layers for Deep Network Architecture
4. Input and Output Encoding
5. Quantum-Classical Hybrid Training
6. Simulation and Evaluation with Qiskit Aer

Dependencies:
- qiskit
- qiskit-aer
- numpy
- scipy
- nltk

Ensure all dependencies are installed before running the script.

====================================================================================================
"""

import sys
import numpy as np
import json
import logging
from typing import List, Dict
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Initialize NLTK data (only the first time)
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================
# Utility Functions
# ============================

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


# ============================
# Quantum Neuron (20-Qubit Logical Unit)
# ============================

class QuantumNeuron:
    """
    A 20-qubit Quantum Neuron implementing logical qubit encoding with error correction.
    """
    def __init__(self, qubits: List[int]):
        """
        Initialize the QuantumNeuron with specified qubit indices.
        
        Parameters:
        qubits (List[int]): List of 20 qubit indices assigned to this neuron.
        """
        if len(qubits) != 20:
            sys.exit("Error: QuantumNeuron requires exactly 20 qubits.")
        self.qubits = qubits
        self.circuit = QuantumCircuit(20)
        self._encode_logical_qubit()
        self._apply_processing()

    def _encode_logical_qubit(self):
        """
        Encode the logical qubit using redundancy and entanglement for error correction.
        """
        # Step 1: Initialize first 10 qubits in superposition
        for i in range(10):
            self.circuit.h(i)
        
        # Step 2: Encode redundancy across next 10 qubits
        for i in range(10):
            self.circuit.cx(i, i + 10)
        
        # Step 3: Add entanglement and parity checks
        for i in range(10):
            self.circuit.cz(i, i + 10)
            self.circuit.cx(i + 10, i)
        
        self.circuit.barrier()

    def _apply_processing(self):
        """
        Apply parameterized rotations for processing information.
        """
        for i in range(20):
            # Example parameterized rotation; in practice, these would be trainable parameters
            self.circuit.rx(0.5, i)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:
        """
        Retrieve the QuantumCircuit representing this neuron.
        
        Returns:
        QuantumCircuit: The quantum circuit of the neuron.
        """
        return self.circuit.copy()


# ============================
# Graphene-Like Layer Structure
# ============================

class GrapheneLayer:
    """
    Graphene-like layer composed of multiple Quantum Neurons arranged in a hexagonal lattice.
    """
    def __init__(self, num_neurons: int, starting_qubit: int):
        """
        Initialize the GrapheneLayer with a specified number of neurons.
        
        Parameters:
        num_neurons (int): Number of Quantum Neurons in this layer.
        starting_qubit (int): The starting qubit index for this layer.
        """
        self.num_neurons = num_neurons
        self.starting_qubit = starting_qubit
        self.qubit_mapping = self._assign_qubits()
        self.circuit = QuantumCircuit(self.num_neurons * 20)
        self._build_layer()

    def _assign_qubits(self) -> List[List[int]]:
        """
        Assign qubits to each neuron in the layer.
        
        Returns:
        List[List[int]]: A list where each element is a list of 20 qubits for a neuron.
        """
        mapping = []
        for n in range(self.num_neurons):
            qubits = list(range(self.starting_qubit + n * 20, self.starting_qubit + (n + 1) * 20))
            mapping.append(qubits)
        return mapping

    def _build_layer(self):
        """
        Construct the GrapheneLayer by adding Quantum Neurons and entangling them.
        """
        # Add Quantum Neurons
        for neuron_qubits in self.qubit_mapping:
            neuron = QuantumNeuron(neuron_qubits)
            self.circuit.compose(neuron.get_circuit(), qubits=range(len(neuron_qubits)), inplace=True)
        
        # Entangle adjacent neurons (string-like connections)
        for i in range(self.num_neurons - 1):
            # Entangle the last qubit of neuron i with the first qubit of neuron i+1
            last_qubit_current = (i + 1) * 20 - 1
            first_qubit_next = (i + 1) * 20
            self.circuit.cz(last_qubit_current, first_qubit_next)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:
        """
        Retrieve the QuantumCircuit representing this GrapheneLayer.
        
        Returns:
        QuantumCircuit: The quantum circuit of the graphene layer.
        """
        return self.circuit.copy()


# ============================
# Stacked Layers for Deep Network
# ============================

class StackedLayers:
    """
    Stack multiple GrapheneLayers to form a deep quantum neural network.
    """
    def __init__(self, num_layers: int, neurons_per_layer: int):
        """
        Initialize the StackedLayers with specified depth and width.
        
        Parameters:
        num_layers (int): Number of GrapheneLayers to stack.
        neurons_per_layer (int): Number of Quantum Neurons per GrapheneLayer.
        """
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.circuit = QuantumCircuit(num_layers * neurons_per_layer * 20)
        self._build_stacked_layers()

    def _build_stacked_layers(self):
        """
        Construct the stacked layers by adding GrapheneLayers and connecting them.
        """
        for layer_num in range(self.num_layers):
            starting_qubit = layer_num * self.neurons_per_layer * 20
            graphene_layer = GrapheneLayer(self.neurons_per_layer, starting_qubit)
            self.circuit.compose(graphene_layer.get_circuit(), qubits=range(starting_qubit, starting_qubit + self.neurons_per_layer * 20), inplace=True)
            
            # Entangle layers if not the first layer
            if layer_num > 0:
                # Entangle the last qubit of the previous layer with the first qubit of the current layer
                prev_layer_last_qubit = (layer_num - 1) * self.neurons_per_layer * 20 + self.neurons_per_layer * 20 - 1
                current_layer_first_qubit = layer_num * self.neurons_per_layer * 20
                self.circuit.cz(prev_layer_last_qubit, current_layer_first_qubit)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:
        """
        Retrieve the QuantumCircuit representing the stacked layers.
        
        Returns:
        QuantumCircuit: The quantum circuit of the stacked layers.
        """
        return self.circuit.copy()


# ============================
# Quantum Language Model (QELM)
# ============================

class QuantumLanguageModel:
    """
    Comprehensive Quantum-Enhanced Language Model combining input encoding, stacked layers, and output measurement.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, neurons_per_layer: int):
        """
        Initialize the QuantumLanguageModel with specified parameters.
        
        Parameters:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding vectors.
        num_layers (int): Number of stacked GrapheneLayers.
        neurons_per_layer (int): Number of Quantum Neurons per GrapheneLayer.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        
        # Initialize embeddings
        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        
        # Initialize quantum layers
        self.stacked_layers = StackedLayers(num_layers, neurons_per_layer)
        
        # Initialize projection layer
        self.W_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01  # Example projection matrix
        
        # Initialize output weights
        self.W_out = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01  # Output matrix
        
        # Initialize quantum parameters (for parameterized gates)
        self._initialize_quantum_params()
    
    def _initialize_quantum_params(self):
        """
        Randomly initialize quantum parameters for training.
        """
        # Example: Initialize rotation angles for RX gates; in practice, these should be trainable
        self.rotation_angles = {}
        for gate_idx in range(self.stacked_layers.circuit.num_qubits):
            self.rotation_angles[gate_idx] = np.random.uniform(0, 2 * np.pi)
    
    def encode_input(self, input_id: int) -> np.ndarray:
        """
        Encode an input token ID into its corresponding embedding vector.
        
        Parameters:
        input_id (int): ID of the input token.
        
        Returns:
        np.ndarray: Embedding vector for the input token.
        """
        if input_id >= self.vocab_size:
            sys.exit(f"Error: Input ID {input_id} exceeds vocabulary size {self.vocab_size}.")
        return self.embeddings[input_id]
    
    def build_full_circuit(self, input_id: int) -> QuantumCircuit:
        """
        Construct the full quantum circuit for a given input token.
        
        Parameters:
        input_id (int): ID of the input token.
        
        Returns:
        QuantumCircuit: The complete quantum circuit for the model.
        """
        # Input Encoding: Prepare the initial state based on the embedding vector
        input_embedding = self.encode_input(input_id)
        input_circuit = QuantumCircuit(self.stacked_layers.circuit.num_qubits)
        
        # Encode embedding into qubits (example: amplitude encoding)
        # Note: Amplitude encoding requires normalization and proper sizing
        # Here, we pad the embedding vector with zeros to match the statevector size
        num_qubits = self.stacked_layers.circuit.num_qubits
        state_prep_length = 2**num_qubits
        if len(input_embedding) > state_prep_length:
            sys.exit("Error: Embedding dimension exceeds statevector size.")
        
        state_prep_vec = np.zeros(state_prep_length, dtype=complex)
        state_prep_vec[:len(input_embedding)] = input_embedding.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        input_circuit.initialize(state_prep_vec, qubits=range(num_qubits))
        
        # Combine with stacked layers
        full_circuit = QuantumCircuit(num_qubits, self.vocab_size)
        full_circuit.compose(input_circuit, inplace=True)
        full_circuit.compose(self.stacked_layers.get_circuit(), inplace=True)
        
        # Apply projection layer (example: simple linear transformation using RX gates)
        for qubit in range(num_qubits):
            angle = self.rotation_angles.get(qubit, 0.0)
            full_circuit.rx(angle, qubit)
        
        full_circuit.barrier()
        
        # Output Layer: Measure qubits to generate logits
        for token in range(self.vocab_size):
            if token < num_qubits:
                full_circuit.measure(token, token)
            else:
                # For tokens beyond the number of qubits, map to the closest qubit
                mapped_qubit = token % num_qubits
                full_circuit.measure(mapped_qubit, token)
        
        return full_circuit
    
    def forward(self, input_id: int) -> np.ndarray:
        """
        Perform a forward pass through the model to obtain logits for the input token.
        
        Parameters:
        input_id (int): ID of the input token.
        
        Returns:
        np.ndarray: Logits for each token in the vocabulary.
        """
        circuit = self.build_full_circuit(input_id)
        backend = Aer.get_backend('aer_simulator')
        noise_model = self._get_noise_model()
        
        # Execute the circuit
        job = execute(circuit, backend, noise_model=noise_model, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to logits
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        for outcome, count in counts.items():
            for idx, bit in enumerate(reversed(outcome)):
                if idx < self.vocab_size:
                    logits[idx] += count if bit == '1' else 0
        return logits
    
    def _get_noise_model(self) -> NoiseModel:
        """
        Define a depolarizing noise model for simulation.
        
        Returns:
        NoiseModel: The noise model to apply during simulation.
        """
        noise_model = NoiseModel()
        depol_error = depolarizing_error(0.01, 1)  # 1% error rate for single-qubit gates
        depol_cx_error = depolarizing_error(0.02, 2)  # 2% error rate for two-qubit gates
        noise_model.add_all_qubit_quantum_error(depol_error, ['rx', 'ry', 'rz', 'h', 'cx', 'cz'])
        noise_model.add_all_qubit_quantum_error(depol_cx_error, ['cx', 'cz'])
        return noise_model
    
    def get_all_parameters(self) -> np.ndarray:
        """
        Retrieve all trainable parameters as a single array.
        
        Returns:
        np.ndarray: Array of all trainable parameters.
        """
        return np.array(list(self.rotation_angles.values()))
    
    def set_all_parameters(self, params: np.ndarray):
        """
        Update all trainable parameters from a single array.
        
        Parameters:
        params (np.ndarray): Array of parameter values.
        """
        if len(params) != len(self.rotation_angles):
            sys.exit("Error: Parameter array length mismatch.")
        for idx, qubit in enumerate(self.rotation_angles.keys()):
            self.rotation_angles[qubit] = params[idx]
    
    def to_dict(self) -> dict:
        """
        Serialize the model parameters to a dictionary.
        
        Returns:
        dict: Serialized model parameters.
        """
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "embeddings": self.embeddings.tolist(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "rotation_angles": self.rotation_angles,
            "version": "1.0"
        }
    
    def from_dict(self, data: dict):
        """
        Load model parameters from a dictionary.
        
        Parameters:
        data (dict): Dictionary containing model parameters.
        """
        if data.get("version") != "1.0":
            sys.exit("Error: Unsupported model version.")
        self.vocab_size = data["vocab_size"]
        self.embed_dim = data["embed_dim"]
        self.num_layers = data["num_layers"]
        self.neurons_per_layer = data["neurons_per_layer"]
        self.embeddings = np.array(data["embeddings"], dtype=np.float32)
        self.W_proj = np.array(data["W_proj"], dtype=np.float32)
        self.W_out = np.array(data["W_out"], dtype=np.float32)
        self.rotation_angles = data["rotation_angles"]
    
    def save_model(self, filepath: str):
        """
        Save the model parameters to a JSON file.
        
        Parameters:
        filepath (str): Path to the file where the model will be saved.
        """
        model_dict = self.to_dict()
        try:
            with open(filepath, 'w') as f:
                json.dump(model_dict, f)
            logging.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            sys.exit(1)
    
    def load_model(self, filepath: str):
        """
        Load the model parameters from a JSON file.
        
        Parameters:
        filepath (str): Path to the file from which the model will be loaded.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.from_dict(data)
            logging.info(f"Model successfully loaded from {filepath}")
        except FileNotFoundError:
            sys.exit(f"Error: File {filepath} not found.")
        except json.JSONDecodeError:
            sys.exit(f"Error: File {filepath} is not a valid JSON file.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            sys.exit(1)


# ============================
# Training Functions
# ============================

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, backend: str = 'aer_simulator') -> float:
    """
    Compute the Mean Squared Error (MSE) loss between model predictions and targets.
    
    Parameters:
    params (np.ndarray): Array of model parameters to update.
    model (QuantumLanguageModel): The QELM instance.
    X (np.ndarray): Input token IDs.
    Y (np.ndarray): Target one-hot vectors.
    backend (str): Qiskit Aer backend to use for simulation.
    
    Returns:
    float: The computed MSE loss.
    """
    # Update model parameters
    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = np.mean((logits - target) ** 2)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.1):
    """
    Train the Quantum Language Model using gradient-based optimization.
    
    Parameters:
    model (QuantumLanguageModel): The QELM instance.
    X (np.ndarray): Input token IDs.
    Y (np.ndarray): Target one-hot vectors.
    epochs (int): Number of training epochs.
    learning_rate (float): Learning rate for parameter updates.
    """
    # Initialize parameters
    params = model.get_all_parameters()
    
    for epoch in range(epochs):
        logging.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        # Define the objective function for this epoch
        objective = lambda p: quantum_loss(p, model, X, Y)
        
        # Perform optimization using SciPy's minimize
        result = minimize(objective, params, method='BFGS')
        
        if not result.success:
            logging.warning(f"Epoch {epoch + 1}: Optimization did not converge.")
        
        # Update parameters
        params = result.x
        
        # Update the model with new parameters
        model.set_all_parameters(params)
        
        # Compute and log the loss
        loss = result.fun
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    logging.info("Training completed.")


# ============================
# Dataset Preparation
# ============================

def create_synthetic_dataset(vocab_size: int, num_samples: int = 100):
    """
    Create a synthetic dataset for demonstration purposes.
    Each sample consists of an input token ID and a target one-hot vector.
    
    Parameters:
    vocab_size (int): Size of the vocabulary.
    num_samples (int): Number of samples to generate.
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Input IDs and target one-hot vectors.
    """
    X = np.random.randint(0, vocab_size, size=(num_samples,))
    Y = np.zeros((num_samples, vocab_size), dtype=np.float32)
    for i in range(num_samples):
        target_id = np.random.randint(0, vocab_size)
        Y[i, target_id] = 1.0
    return X, Y

def load_real_dataset(file_path: str, vocab_size: int):
    """
    Load and preprocess a real language dataset.
    
    Parameters:
    file_path (str): Path to the text file containing the dataset.
    vocab_size (int): Number of top tokens to include in the vocabulary.
    
    Returns:
    Tuple[np.ndarray, np.ndarray, Dict[str, int]]: Input IDs, target one-hot vectors, and token-to-ID mapping.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        sys.exit(f"Error reading the dataset file: {e}")
    
    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    
    # Select top vocab_size tokens
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    
    # Convert tokens to IDs
    X = []
    Y = []
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        if current_token in token_to_id and next_token in token_to_id:
            X.append(token_to_id[current_token])
            Y.append(token_to_id[next_token])
    
    # One-hot encode targets
    Y_one_hot = np.zeros((len(Y), vocab_size), dtype=np.float32)
    for i, target_id in enumerate(Y):
        Y_one_hot[i, target_id] = 1.0
    
    return np.array(X), Y_one_hot, token_to_id


# ============================
# Simulation and Evaluation
# ============================

def simulate_and_evaluate(model: QuantumLanguageModel, input_id: int, target_id: int):
    """
    Simulate the model for a single input and evaluate fidelity against the target.
    
    Parameters:
    model (QuantumLanguageModel): The QELM instance.
    input_id (int): ID of the input token.
    target_id (int): ID of the target token.
    
    Returns:
    float: Fidelity score between the simulated state and the target state.
    """
    # Build and simulate the circuit
    circuit = model.build_full_circuit(input_id)
    backend = Aer.get_backend('aer_simulator')
    noise_model = model._get_noise_model()
    job = execute(circuit, backend, noise_model=noise_model, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Generate ideal state
    ideal_circuit = QuantumCircuit(circuit.num_qubits, model.vocab_size)
    input_embedding = model.encode_input(input_id)
    normalized_embedding = normalize_vector(input_embedding)
    state_prep_length = 2**model.stacked_layers.circuit.num_qubits
    if len(normalized_embedding) > state_prep_length:
        sys.exit("Error: Embedding dimension exceeds statevector size.")
    state_prep_vec = np.zeros(state_prep_length, dtype=complex)
    state_prep_vec[:len(normalized_embedding)] = normalized_embedding.astype(complex)
    state_prep_vec = normalize_vector(state_prep_vec)
    ideal_circuit.initialize(state_prep_vec, qubits=range(model.stacked_layers.circuit.num_qubits))
    ideal_circuit.barrier()
    ideal_circuit.compose(model.stacked_layers.get_circuit(), inplace=True)
    
    # Apply projection layer with ideal parameters (no rotation)
    for qubit in range(model.stacked_layers.circuit.num_qubits):
        ideal_circuit.rx(0.0, qubit)  # No rotation for ideal state
    
    ideal_circuit.barrier()
    
    # Output Layer: Measure qubits to generate logits
    for token in range(model.vocab_size):
        if token < model.stacked_layers.circuit.num_qubits:
            ideal_circuit.measure(token, token)
        else:
            # For tokens beyond the number of qubits, map to the closest qubit
            mapped_qubit = token % model.stacked_layers.circuit.num_qubits
            ideal_circuit.measure(mapped_qubit, token)
    
    # Simulate the ideal circuit
    ideal_job = execute(ideal_circuit, backend, shots=1024)
    ideal_result = ideal_job.result()
    ideal_counts = ideal_result.get_counts()
    
    # Calculate fidelity
    target_bitstring = '0' * model.stacked_layers.circuit.num_qubits
    model_fidelity = state_fidelity(counts.get(target_bitstring, 0) / 1024, ideal_counts.get(target_bitstring, 0) / 1024)
    
    logging.info(f"Fidelity for input ID {input_id} targeting ID {target_id}: {model_fidelity:.4f}")
    return model_fidelity


# ============================
# Main Execution
# ============================

def main():
    """
    Main function to initialize the QELM, prepare data, train, and evaluate.
    """
    try:
        # Define model parameters for a manageable simulation
        vocab_size = 16           # Reduced vocabulary size for testing
        embed_dim = 4             # Reduced embedding dimension
        num_layers = 1            # Start with one layer
        neurons_per_layer = 2     # Fewer neurons per layer
        epochs = 2                # Fewer training epochs for quick testing
        learning_rate = 0.1
        
        # Initialize the Quantum Language Model
        qlm = QuantumLanguageModel(vocab_size, embed_dim, num_layers, neurons_per_layer)
        
        # Prepare dataset (using synthetic data for demonstration)
        X, Y = create_synthetic_dataset(vocab_size, num_samples=10)  # Fewer samples
        
        # Train the model
        train_quantum_model(qlm, X, Y, epochs=epochs, learning_rate=learning_rate)
        
        # Save the trained model
        qlm.save_model("quantum_llm.qelm")
        
        # Evaluate the model on a sample input
        sample_input_id = X[0]
        sample_target_id = np.argmax(Y[0])
        fidelity = simulate_and_evaluate(qlm, sample_input_id, sample_target_id)
        print(f"Sample Fidelity: {fidelity:.4f}")
        
        # Load the model back (demonstration)
        qlm_loaded = QuantumLanguageModel(vocab_size, embed_dim, num_layers, neurons_per_layer)
        qlm_loaded.load_model("quantum_llm.qelm")
        
        # Re-evaluate the loaded model
        fidelity_loaded = simulate_and_evaluate(qlm_loaded, sample_input_id, sample_target_id)
        print(f"Sample Fidelity After Loading: {fidelity_loaded:.4f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    finally:
        # Pause before exiting
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
