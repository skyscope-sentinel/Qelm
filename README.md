# Quantum-Enhanced Language Model (QELM)

![License](https://img.shields.io/github/license/R-D-BioTech-Alaska/QELM)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.3.0-orange)
![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)
![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/QELM?style=social)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Benefits Over Traditional LLMs](#benefits-over-traditional-llms)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
    - [With Synthetic Data](#with-synthetic-data)
    - [With a Real Dataset](#with-a-real-dataset)
  - [Performing Inference](#performing-inference)
  - [Viewing Help/Usage Information](#viewing-helpusage-information)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)
- [Contact](#contact)

## Overview

Welcome to the **Quantum-Enhanced Language Model (QELM)** – an innovative project that merges the power of quantum computing with natural language processing to create a next-generation language model. Leveraging **Qiskit** and **Qiskit Aer**, QELM explores the potential of quantum circuits in enhancing language understanding and generation capabilities.

![QELM Diagram](docs/images/QELM_Diagram.png) <!-- Replace with actual image path -->

## Features

- **Quantum Parameter Optimization:** Utilizes the Parameter Shift Rule for gradient-based optimization within quantum circuits.
- **Advanced Quantum Circuits:** Implements entangling gates and multiple layers for complex state manipulations.
- **Synthetic and Real Dataset Support:** Capable of training on both synthetic datasets for testing and real-world language data.
- **Enhanced Model Architecture:** Incorporates residual connections and normalization for stable and efficient training.
- **Parameter Persistence:** Robust saving and loading mechanisms with versioning to ensure model integrity.
- **User-Friendly CLI:** Intuitive command-line interface for training, inference, and model management.

## Benefits Over Traditional LLMs

### 1. **Quantum Parallelism**
Quantum computers can process a vast number of states simultaneously due to superposition and entanglement, potentially enabling faster and more efficient computations compared to classical models.

### 2. **Enhanced Representational Capabilities**
Quantum states can represent complex, high-dimensional data compactly, allowing QELM to capture intricate patterns and relationships in language data more effectively.

### 3. **Improved Optimization**
Quantum algorithms can tackle optimization problems differently, potentially finding better minima or converging faster during training.

### 4. **Parameter Efficiency**
Quantum circuits can represent complex functions with fewer parameters, leading to more efficient models that are less resource-intensive to store and deploy.

### 5. **Novel Learning Paradigms**
Integrating quantum principles inspires new model architectures, offering unique advantages in handling language tasks and potentially mimicking human-like language understanding more closely.

## Installation

### Prerequisites

- **Python 3.7 to 3.11**
- **Git** installed on your machine

### Clone the Repository

```bash
git clone https://github.com/R-D-BioTech-Alaska/QELM.git
cd QELM
```

### Setup Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'qiskit_env'
python -m venv qiskit_env

# Activate the virtual environment

# Windows
qiskit_env\Scripts\activate

# Unix/Linux/MacOS
source qiskit_env/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter permission issues, you can add the `--user` flag to the `pip install` command.

## Usage

QELM provides a command-line interface (CLI) to facilitate training, inference, saving, and loading models.

### Training the Model

You can train QELM using either synthetic data (for testing purposes) or a real dataset.

#### With Synthetic Data

Synthetic data is useful for initial testing and ensuring that the training pipeline works correctly.

```bash
python Qelm2.py --train --epochs 20 --lr 0.05
```

**Parameters:**

- `--train`: Initiates the training process.
- `--epochs 20`: Sets the number of training epochs to 20.
- `--lr 0.05`: Sets the learning rate to 0.05.
- `--save_path`: (Optional) Path to save the trained model (default: `quantum_llm_model_enhanced.json`).
- `--dataset_path`: (Optional) Path to a real dataset file. If not provided, synthetic data will be used.

**Expected Output:**

```plaintext
INFO:root:Creating synthetic dataset...
INFO:root:Starting training...
INFO:root:Starting Epoch 1/20
INFO:root:Epoch 1/20, Average Loss: 0.012345
INFO:root:Starting Epoch 2/20
INFO:root:Epoch 2/20, Average Loss: 0.011234
...
INFO:root:Starting Epoch 20/20
INFO:root:Epoch 20/20, Average Loss: 0.010123
INFO:root:Training completed.
INFO:root:Model saved to quantum_llm_model_enhanced.json
```

#### With a Real Dataset

If you have a real language dataset, you can specify its path using the `--dataset_path` argument. Ensure that your dataset is in plain text format.

```bash
python Qelm2.py --train --epochs 20 --lr 0.05 --dataset_path path_to_your_dataset.txt
```

**Parameters:**

- `--train`: Initiates the training process.
- `--epochs 20`: Sets the number of training epochs to 20.
- `--lr 0.05`: Sets the learning rate to 0.05.
- `--dataset_path path_to_your_dataset.txt`: Specifies the path to your real dataset file.
- `--save_path`: (Optional) Path to save the trained model (default: `quantum_llm_model_enhanced.json`).

**Expected Output:**

```plaintext
INFO:root:Loading real dataset...
INFO:root:Starting training...
INFO:root:Starting Epoch 1/20
INFO:root:Epoch 1/20, Average Loss: 0.012345
INFO:root:Starting Epoch 2/20
INFO:root:Epoch 2/20, Average Loss: 0.011234
...
INFO:root:Starting Epoch 20/20
INFO:root:Epoch 20/20, Average Loss: 0.010123
INFO:root:Training completed.
INFO:root:Model saved to quantum_llm_model_enhanced.json
```

**Notes:**

- **Dataset Preparation:** Ensure your dataset file (`path_to_your_dataset.txt`) is properly formatted and preprocessed. The script tokenizes the text and maps the most frequent tokens to unique IDs based on the specified `vocab_size`.
- **Vocabulary Size:** The default `vocab_size` is set to 256. Adjust this in the script if your dataset requires a different size.

### Performing Inference

After training the model, you can perform inference to generate logits (predicted outputs) based on an input token ID.

```bash
python Qelm2.py --inference --input_id 5 --load_path quantum_llm_model_enhanced.json
```

**Parameters:**

- `--inference`: Initiates the inference process.
- `--input_id 5`: Specifies the input token ID for which you want to generate logits.
- `--load_path quantum_llm_model_enhanced.json`: Specifies the path to the saved model file.

**Expected Output:**

```plaintext
Logits: [0.00123456 0.00234567 -0.00012345 0.00345678 ...]
```

**Notes:**

- **Input ID Validation:** Ensure that the `input_id` provided is within the range of your vocabulary size (e.g., 0 to 255 for `vocab_size=256`). Providing an ID outside this range will result in an error.
- **Interpreting Logits:** The output logits represent the model's predictions for each token in the vocabulary. These can be further processed (e.g., using a softmax function) to obtain probability distributions over the vocabulary.

### Viewing Help/Usage Information

To understand all available options and ensure you're using the script correctly:

```bash
python Qelm2.py --help
```

**Expected Output:**

```plaintext
usage: Qelm2.py [-h] [--train] [--inference] [--input_id INPUT_ID]
               [--save_path SAVE_PATH] [--load_path LOAD_PATH]
               [--epochs EPOCHS] [--lr LR] [--dataset_path DATASET_PATH]

Quantum-Enhanced Language Model (QELM) - Enhanced Version

optional arguments:
  -h, --help            show this help message and exit
  --train               Train the model
  --inference           Run inference
  --input_id INPUT_ID   Input token ID for inference
  --save_path SAVE_PATH
                        Path to save the model
  --load_path LOAD_PATH
                        Path to load the model
  --epochs EPOCHS       Number of training epochs
  --lr LR               Learning rate
  --dataset_path DATASET_PATH
                        Path to real dataset file (optional)
```

## Project Structure

```plaintext
QELM/
├── Qelm2.py
├── requirements.txt
├── README.md
├── quantum_llm_model_enhanced.json
└── docs/
    └── images/
        └── QELM_Diagram.png
```

- **Qelm2.py:** The main script containing the Quantum-Enhanced Language Model implementation.
- **requirements.txt:** Lists all Python dependencies required to run QELM.
- **README.md:** This documentation file.
- **quantum_llm_model_enhanced.json:** Saved model parameters after training.
- **docs/images/QELM_Diagram.png:** Diagram illustrating the QELM architecture (replace with actual image).

## Credits

Developed by **Brenton Carter** (Inserian).

Special thanks to the [Qiskit](https://qiskit.org/) community for providing robust quantum computing tools and resources that made this project possible.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, suggestions, or contributions, feel free to reach out:

- **Email:** [contact@rdbiotechalaska.com](mailto:contact@rdbiotechalaska.com)
- **GitHub:** [R-D-BioTech-Alaska](https://github.com/R-D-BioTech-Alaska)
- **Website:** [RDBioTech.org](http://RDBioTech.org)

---

> **Disclaimer:** Quantum computing is an emerging field, and this project serves as an experimental exploration into integrating quantum circuits with language modeling. While promising, it is subject to the current limitations of quantum hardware and algorithms.
```

🚀🔬✨
