# Quantum-Enhanced Language Model (QELM) – Theoretical

[![License](https://img.shields.io/github/license/R-D-BioTech-Alaska/QELM)](LICENSE)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.3.0-orange)
![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)
![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/QELM?style=social)

**QELM** (Quantum-Enhanced Language Model) merges **quantum computing** and **NLP** to provide compact-yet-powerful language models with advanced features like **multi-block** quantum transformers, **ring entanglement**, data reuploading, parameter-shift gradient training, and much more.  

> **Important**:  `QelmT.py` is the **new** consolidated training/inference script. The older scripts (`Qelm2.py`, `QelmGUI.py`) remain functional and can still be used. However, they are now considered **outdated**.

---

## Table of Contents
1. [What’s New in QelmT.py?](#whats-new-in-qelmtpy)
2. [Quantum vs. Classical Size Comparison](#quantum-vs-classical-size-comparison)
3. [Features](#features)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Cloning the Repository](#cloning-the-repository)
   - [Virtual Environment Setup](#virtual-environment-setup)
   - [Dependency Installation](#dependency-installation)
5. [Usage with QelmT.py (Recommended)](#usage-with-qelmtpy-recommended)
   - [Basic Command Line Training](#basic-command-line-training)
   - [Performing Inference](#performing-inference)
   - [Advanced Options](#advanced-options)
6. [(Outdated but Working) Legacy Scripts](#outdated-but-working-legacy-scripts)
   - [Qelm2.py](#qelm2py)
   - [QelmGUI.py](#qelmgui)
   - [QELMChatUI.py](#qelmchatui)
7. [Project Structure](#project-structure)
8. [License](#license)
9. [Contact](#contact)

---

## What’s New in QelmT.py?

**QelmT.py** is the newest codebase that includes better control, sub-bit encoding and entropy control.
- **Training** with either real or synthetic datasets
- **Parameter tuning** (learning rate, epochs, advanced quantum ansatz, multi-threading, data re-uploading, sub-bit encoding, entropy factor, etc.)
- **Inference** (prompt-based generation and conversation)
- **Resource monitoring** (CPU/GPU usage)  
- **Model checkpointing** (save & load using a `.qelm` file)

---

## Quantum vs. Classical Size Comparison

With the addition of **sub-bit encoding** and **entropy-based qubit mixing**, QELM has become even more space-efficient than our earlier comparisons indicated. While the original table below provides a rough idea of how QELM’s quantum “compression” compares to typical classical LLMs, **these figures may be **underselling** QELM’s true potential**. In recent tests, leveraging sub-bit encoding at around **13.69 bytes per qubit** and carefully tuned entropy factors in the training phase allowed us to **store more representational information in fewer qubits**, further shrinking the model size.

> **Note**: We retain the original table for historical/contextual reference. If anything, real-world deployments of sub-bit + entropy-optimized QELM will likely show even **greater** size reductions.

| Classical Size (MB) | Classical LLM (bits)      | QELM (bits)               | Relationship   |
|---------------------|---------------------------|---------------------------|----------------|
| 1 MB                | ~8.39×10<sup>6</sup>      | ~8.44×10<sup>7</sup>      | QELM >> LLM    |
| 5 MB                | ~4.19×10<sup>7</sup>      | ~9.84×10<sup>7</sup>      | QELM > LLM     |
| 16.6 MB             | ~1.39×10<sup>8</sup>      | ~1.39×10<sup>8</sup>      | QELM ≈ LLM     |
| 50 MB               | ~4.19×10<sup>8</sup>      | ~2.56×10<sup>8</sup>      | QELM << LLM    |
| 100 MB              | ~8.39×10<sup>8</sup>      | ~4.31×10<sup>8</sup>      | QELM << LLM    |
| 1 GB                | ~8.59×10<sup>9</sup>      | ~3.67×10<sup>9</sup>      | QELM << LLM    |
| 100 GB              | ~8.59×10<sup>11</sup>     | ~3.59×10<sup>11</sup>     | QELM << LLM    |

> With **entanglement**, **sub-bit encoding**, and **entropy-mixed gates**, QELM drastically reduces storage requirements. In practice, you can expect **significantly smaller** footprints than the ones listed here once you enable these advanced techniques.

---

## Features
- **Quantum Circuit Transformers**  
  - Advanced ring entanglement, data reuploading, multi-block attention
  - Parameter-shift gradient training (supports multi-threading)
- **QelmT.py**: One script for everything (training + inference + more)
- **Live Resource Monitoring**  
  - CPU usage, GPU usage if available
- **Lightweight Models**  
  - Potentially 10-100x smaller than classical LLMs of similar capacity currently
- **Wide Range of Tokenization**  
  - Exponential subword, BPE, WordPiece, dynamic vocab, etc.

---

## Installation

### Prerequisites
- **Python 3.7+** (tested up to 3.11)
- **Qiskit** + **Qiskit Aer**
- **NumPy**, **TensorFlow**  
- **Tkinter** (usually included in Python)
- **psutil** (optional, for CPU usage)
- **nltk** (for tokenizing text data)

### Cloning the Repository
```bash
git clone https://github.com/R-D-BioTech-Alaska/QELM.git
cd QELM
```

### Virtual Environment Setup
```bash
python -m venv qiskit_env
# Activate virtualenv:
# Linux/Mac:
source qiskit_env/bin/activate
# Windows:
qiskit_env\Scripts\activate
```

### Dependency Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage with QelmT.py (Recommended)

Below are **basic** usage examples for `QelmT.py`. For more advanced options, use `--help`:

### Basic Command Line Training
```bash
python QelmT.py --train \
                --dataset /path/to/data.txt \
                --vocab_size 8000 \
                --embed_dim 256 \
                --num_heads 4 \
                --hidden_dim 512 \
                --epochs 5 \
                --lr 0.001
```
**Flags**:
- `--train` : Activates training mode
- `--dataset` : Path to your `.txt` dataset
- `--vocab_size` : Limit for vocabulary
- `--embed_dim` : Embedding dimension (must be divisible by `--num_heads`)
- `--num_heads` : Number of attention heads
- `--hidden_dim` : Hidden dimension for feed-forward
- `--epochs` : Number of training epochs
- `--lr` : Learning rate

### Performing Inference
```bash
python QelmT.py --inference \
                --input_token "hello" \
                --max_length 50 \
                --temperature 1.0 \
                --model /path/to/saved_model.qelm
```
**Flags**:
- `--inference` : Inference mode
- `--input_token` : Starting word or token
- `--max_length` : Maximum output tokens
- `--temperature` : Sampling temperature (higher => more random)
- `--model` : Path to a `.qelm` checkpoint

### Advanced Options
- `--num_blocks N` : Multi-block quantum transformers (default=1)
- `--use_advanced_ansatz` : Enable advanced quantum gates
- `--use_data_reuploading` : Use data reuploading technique
- `--sim_method [cpu|gpu|both|simulation]` : Simulation approach
- `--threads N` : For multi-threaded parameter-shift
- `--decimal_precision N` : Force quantum channels to round to N decimals
- `--use_subbit_encoding` : Sub-bit encoding to store more info per qubit

**Check all available flags**:
```bash
python QelmT.py --help
```

---

## (Outdated but Working) Legacy Scripts

We continue to include the older scripts for users who wish to see or compare the original QELM approach. They are **still functional** but are **no longer actively updated**.

### Qelm2.py
A simple command-line script for:
- **Training** (`--train`)
- **Inference** (`--inference`)
- Basic model save/load

### QelmGUI.py
A Tkinter-based GUI with:
- Dataset selection, training hyperparams, real-time logs & progress bars
- Inference tab for text generation

### QELMChatUI.py
Chat-like interface (a la ChatGPT style):
- Multi-turn conversation with QELM
- Model selection, load/save, conversation logs

> **Note**: Both `QelmGUI.py` and `QELMChatUI.py` require a local Python environment with Tkinter.

---

## Project Structure
```
QELM/
├── QelmT.py                # NEW: Unified training+inference script (Recommended)
├── Qelm2.py                # Legacy CLI script
├── QelmGUI.py              # Legacy GUI for training & inference
├── QELMChatUI.py           # Legacy Chat UI
├── requirements.txt
├── docs/
│   └── images/
│       ├── QELM_Diagram.png
│       ├── quantum.png
│       └── Qelm.png
└── README.md               # This documentation
```

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact
For additional guidance, collaboration, or bug reports:
- **Email**: [contact@rdbiotechalaska.com](mailto:contact@rdbiotechalaska.com)
- **GitHub**: [R-D-BioTech-Alaska](https://github.com/R-D-BioTech-Alaska)
- **Website**: [RDBioTech.org](http://RDBioTech.org)  
  <sub>(*Disclaimer: QELM is experimental; community feedback is greatly appreciated.*)</sub>
