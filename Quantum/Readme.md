# QELM (Quantum-Enhanced Language Model) - Quantum - README

Welcome to the **Quantum-Enhanced Language Model** (QELM) - Quantum Project. This repository integrates QELM into IBM's QPU's for testing purposes. You must have an IBM Quantum account to use an API for this.

**Warning** This is not QELM. This has been modified to work with IBM's QPU's to process and understand information. It can create Qelm's given the correct parameters using QPU's, but as of this moment, this is extreme **Beta** 

## Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Project Structure](#project-structure)  
4. [Prerequisites](#prerequisites)  
5. [Installation and Setup](#installation-and-setup)  
6. [Running the GUI Application](#running-the-gui-application)  
7. [Detailed Code Explanation](#detailed-code-explanation)  
   - [AdamOptimizer](#adamoptimizer)  
   - [QuantumLayerBase](#quantumlayerbase)  
   - [QuantumAttentionLayer](#quantumattentionlayer)  
   - [QuantumFeedForwardLayer](#quantumfeedforwardlayer)  
   - [QuantumLanguageModel](#quantumlanguagemodel)  
   - [Dataset Utilities](#dataset-utilities)  
   - [Training Utilities](#training-utilities)  
   - [Inference Utilities](#inference-utilities)  
   - [QELM_GUI (Tkinter GUI)](#qelm_gui)  
8. [Using IBM Quantum Hardware](#using-ibm-quantum-hardware)  
9. [Troubleshooting](#troubleshooting)  
10. [License](#license)  
11. [Contributing](#contributing)  

---

## Overview

The **Quantum-Enhanced Language Model (QELM) - Quantum ** attempts to fuse the concepts of quantum computing with classical deep learning approaches for language modeling. While the quantum functionality here is relatively basic (e.g., employing Qiskit simulators and optional IBM Quantum hardware for certain circuit executions), it demonstrates how quantum circuits could be embedded into attention and feed-forward layers.

You need to understand IBM's Quantum Dashboard before using this. Currently only small calls are allowed through embedding and hidden as it scales qubits exponentially. Heron will be released in the near future with additional Qubits to run these tasks. Until that time this will be an off and on project.

Results should be : Object

Description should be : Runner Result

If a run doesn't go as expected then you may have to lower dimensions to decrease decoherence. Currently we have methods to defuse this but have not yet been incorporated.

**Might experience backend issues when they update the QPU's** This can be fixed by updated calls.

This project includes:
- A simplified language model architecture with quantum layers.
- An **AdamOptimizer** for parameter updates.
- A **Tkinter GUI** for managing training, logging progress, and running inference interactively.
- Functions to generate synthetic data or load real data from text files and tokenize with **NLTK**.
- Optional usage of **IBM Quantum** hardware via QiskitRuntimeService (requires your IBM Quantum API token).

---

## Key Features
1. **Hybrid Quantum-Classical Layers**:  
   - The `QuantumAttentionLayer` and `QuantumFeedForwardLayer` encapsulate quantum circuit creation and execution.
2. **Pluggable Simulation Backends**:  
   - Easily switch between the local `AerSimulator` and IBM Quantum hardware by providing an IBM Quantum API token.
3. **Tkinter GUI**:  
   - Start, stop, and monitor training in real-time.  
   - Save and load models.  
   - Run token-based text generation (inference).  
4. **Parameter-Shift Gradient Computation**:  
   - Demonstrates a naive parameter-shift rule for computing gradients in parallel.
5. **Evaluation Metrics**:  
   - Cross-entropy loss, perplexity, BLEU score (simplistic).
6. **Logging**:  
   - Comprehensive logging with Python’s built-in logging module.

---

## Project Structure

```
qelm/
│
├── README.md                <- You are here!
├── Qelm-Quantum.py          <- Main code containing all classes and the GUI application.
├── requirements.txt         <- Python dependencies.
├── qelm.log                 <- Log file (generated at runtime).
└── ...
```

- **`Qelm-Quantum.py`**  
  This single-file program includes:
  - **Classes**: `QuantumLanguageModel`, `QuantumAttentionLayer`, `QuantumFeedForwardLayer`, `AdamOptimizer`, etc.  
  - **GUI**: The entire `QELM_GUI` class which runs the training and inference interface.  
  - **Utility Functions**: Functions for dataset creation/processing, training loops, gradient computation, etc.
- **`requirements.txt`**  
  Lists primary dependencies needed to run the application.

---

## Prerequisites

1. **Python 3.8+** (Generally recommended; may work on earlier versions but not tested.)
2. **Pip** (or your favorite Python package manager).
3. (Optional) **IBM Quantum Account** if you want to run on real quantum hardware. Sign up at [IBM Quantum](https://quantum-computing.ibm.com/).

---

## Installation and Setup

1. **Clone the Repository**  
   ```bash
   git clone (https://github.com/R-D-BioTech-Alaska/Qelm/new/main/Quantum.git
   cd Qelm-Quantum.py
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   This should install:
   - `qiskit`
   - `qiskit_aer`
   - `qiskit_ibm_runtime`
   - `nltk`
   - `numpy`
   - `psutil` (optional, for CPU usage)
   - and other needed libraries (e.g., `concurrent.futures`, etc.).

3. **Verify NLTK Tokenizers**  
   - The code automatically checks and downloads the `punkt` tokenizer if missing. Make sure you have an internet connection the first time you run it.

4. **(Optional) IBM Quantum Setup**  
   - If you intend to use IBM Quantum hardware, sign up at [IBM Quantum](https://quantum-computing.ibm.com/), retrieve your **API token**, and paste it into the GUI when prompted.

---

## Running the GUI Application

1. **Navigate to the project folder**:
   ```bash
   cd qelm
   ```
2. **Start the GUI**:
   ```bash
   python qelm.py
   ```
3. **GUI Controls**:
   - **Select Dataset**: Choose a text file or proceed with synthetic data.  
   - **Set Hyperparameters**: (Vocabulary size, embedding dimension, etc.)  
   - **Select Backend**: `Simulator` or `IBM Quantum`. If using IBM Quantum, provide your API token.  
   - **Start Training**: Observe real-time logging, CPU usage, training progress, perplexity, BLEU score.  
   - **Stop (Graceful)**: Waits until the current epoch finishes.  
   - **HARD STOP**: Immediately kills the application.  
   - **Save Model / Load Model**: For persisting or restoring a trained model.  
   - **Run Inference**: Enter a token to generate text from the trained model.  
   - **Manage Token Mappings**: Load or view the mappings from tokens to IDs.

---

## Detailed Code Explanation

Below is an overview of the primary classes and functions in `qelm.py`. For thorough understanding, read the inline docstrings in the code.

### AdamOptimizer
- **Location**: Near the top of the file.  
- **Purpose**: Implements a simple Adam optimizer for parameter updates.  
- **Key Methods**:
  - `step(gradients: np.ndarray) -> np.ndarray`:  
    Applies Adam update rule to the stored parameters and returns the new parameter array.

### QuantumLayerBase
- **Location**: Parent class for quantum layers (`QuantumAttentionLayer` and `QuantumFeedForwardLayer`).  
- **Purpose**:  
  - Initializes **Qiskit** `AerSimulator` or **IBM Quantum** runtime service.  
  - Defines a `simulate` method to run circuits on the chosen backend.  
- **Key Features**:  
  - `initialize_simulator()`: Sets up an `AerSimulator` with a configurable `max_parallel_threads`.  
  - `initialize_service()`: Connects to IBM Quantum with your API token.  
  - `simulate(circuit: QuantumCircuit) -> np.ndarray`: Runs or samples from the circuit depending on the backend.

### QuantumAttentionLayer
- **Inherits**: `QuantumLayerBase`  
- **Purpose**: Represents a quantum-attention mechanism.  
- **Key Method**:
  - `forward(x: List[int], mode: str = 'query') -> np.ndarray`:  
    Creates a sample quantum circuit, applies gates, measures, and returns the result (either statevector or sampling distribution).

### QuantumFeedForwardLayer
- **Inherits**: `QuantumLayerBase`  
- **Purpose**: Acts like a feed-forward layer but uses quantum circuits under the hood.  
- **Key Method**:
  - `forward(x: List[int], mode: str = 'w1') -> np.ndarray`:  
    Similar to `QuantumAttentionLayer`, builds a circuit, applies gates, measures, and returns results.

### QuantumLanguageModel
- **Purpose**:  
  - Combines embeddings, quantum attention, and quantum feed-forward into a toy language model.  
  - Demonstrates a single quantum attention head and quantum feed-forward transform.  
- **Key Methods**:
  - `forward(x: List[int], use_residual: bool = False) -> np.ndarray`:  
    Applies embeddings -> quantum attention -> quantum feed-forward -> projection -> output logits.  
  - `train_model(...)`: (Defined in the utility function, not inside this class, see below).  
  - `save_model` / `load_model`: Serialize and deserialize model parameters.  
  - `get_all_parameters` / `set_all_parameters`: Flatten or reshape all model parameters for convenient updates.

### Dataset Utilities
- **`create_synthetic_dataset(vocab_size: int, num_samples: int = 500)`:**  
  Creates random integer tokens as X (inputs) and Y (targets).
- **`load_real_dataset(file_path: str, vocab_size: int)`:**  
  Uses `nltk.word_tokenize`, builds a frequency-based vocabulary, returns tokenized sequences (X, Y) and the mapping dictionary.

### Training Utilities
- **`train_model(...)`:**  
  - Orchestrates the training loop over epochs.  
  - Uses `compute_gradients_parallel` to get parameter-shift gradients.  
  - Supports real-time logging updates via a queue.  
- **`compute_gradient_for_parameter(...)`:**  
  - The heart of the parameter-shift rule for a single parameter.  
- **`compute_gradients_parallel(...)`:**  
  - Spawns multiple processes for computing gradients in parallel.  
  - Aggregates results back into a single gradient vector.

### Inference Utilities
- **`run_inference(...)`:**  
  - Given an initial token (or tokens), repeatedly generates new tokens by sampling from the model’s output distribution.  
  - Uses optional temperature scaling for randomness.

### QELM_GUI
- **Purpose**:  
  - A fully functional **Tkinter** GUI for training and inference.  
  - Offers user-friendly controls for hyperparameter selection, dataset loading, result logging, etc.  
- **Key Sections**:  
  - **Dataset Selection**: Let users pick a text file or default to synthetic.  
  - **Hyperparameters**: Set vocabulary size, embedding dim, etc.  
  - **Execution Settings**: Choose between local simulator and IBM Quantum.  
  - **Progress Bars**: Shows training progress and gradient computation progress.  
  - **Real-time Logging**: Outputs training details to a scrolled text box.  
  - **Inference**: Generate responses from an input token.  
  - **Token Management**: Load or display token mappings.

---

## Using IBM Quantum Hardware

1. **Select "IBM Quantum"** in the **Train Model** tab under "Execution Backend."  
2. **Provide your IBM Quantum API token**.  
3. Once the training starts, the quantum circuits from `QuantumAttentionLayer` and `QuantumFeedForwardLayer` will execute on your chosen backend.  
4. Note that performance will be significantly slower on real hardware or cloud simulators, and queue times may apply.

**Important**: By default, the code references `'ibmq_qasm_simulator'` as a preferred backend. You can modify `preferred_backend` in `initialize_backend` as desired.  

---

## Troubleshooting

1. **Qiskit Import Error**  
   - Make sure `qiskit`, `qiskit_aer`, and `qiskit_ibm_runtime` are installed. Check the output of `pip list`.  

2. **NLTK Tokenizer Not Found**  
   - The code attempts to auto-download the `'punkt'` tokenizer if missing. Ensure internet access is available on the first run or manually install NLTK data:  
     ```python
     import nltk
     nltk.download('punkt')
     ```
3. **IBM Quantum Errors**  
   - Verify your `ibmq_token` is correct and has not expired.  
   - Confirm your IBM Quantum account details via `QiskitRuntimeService` if you get authentication errors.  

4. **Parameter-Shifting Slow**  
   - The naive parameter-shift rule can be slow for large models. Consider reducing the model size or the number of samples.  

5. **GUI Freezes**  
   - Python’s **Tkinter** can freeze if the main thread is blocked. We use multi-threading for the training and `ProcessPoolExecutor` for gradient computations. If you encounter issues, ensure your system can handle the parallel processes.

---

## License

This project is distributed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use and modify it for your own projects. 

---

## Contributing

We welcome contributions! If you find issues or want to extend functionality:
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit your changes and push to your fork.
4. Open a Pull Request detailing your changes.

Thank you for your interest in improving QELM!

---

**Happy Quantum-Coding!** If you have any questions or encounter any issues, feel free to open an issue on GitHub or drop a message.  
