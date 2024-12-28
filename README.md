# Quantum-Enhanced Language Model (QELM) (QLM)

![License](https://img.shields.io/github/license/R-D-BioTech-Alaska/QELM)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.3.0-orange)
![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)
![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/QELM?style=social)

## Table of Contents
- [Overview](#overview)
- [Comparison with Regular LLMs](#comparison-with-regular-llms)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Performing Inference](#performing-inference)
  - [Graphical Interfaces](#graphical-interfaces)
    - [1. QelmGUI (Training + Inference)](#1-qelmgui-training--inference)
    - [2. QELMChatUI (Conversational UI)](#2-qelmchatui-conversational-ui)
  - [Legacy Command Line (Older Script)](#legacy-command-line-older-script)
  - [Viewing Help/Usage](#viewing-helpusage)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)
- [Contact](#contact)

---

## Overview
The **Quantum-Enhanced Language Model (QELM)** merges quantum computing with natural language processing to produce extremely **compact** yet powerful language models. By encoding token embeddings into **quantum states** and leveraging **entanglement**, QELM drastically reduces storage requirements compared to classical LLMs. This makes QELM an excellent choice for edge devices or memory-limited environments.

---

## Comparison with Regular LLMs
Classical LLMs often reach **6 - 60 GB** (or more) even for modest architectures. In current comparison tests QELM, by contrast, typically yields models around **2 MB** when a classical llm would be **15-20 MB, delivering:
- **8–9x size reduction**  
- Similar perplexity/performance (e.g., perplexity ~100)  
- Efficient parameter usage through quantum ansätze and ring entanglement  

In short, quantum-based “compression” can significantly reduce overhead without compromising on capabilities.

---

## Features
- **Sophisticated Quantum Circuits**  
  - **Advanced Ansatz**: RY, RZ, ring entanglement patterns, optional data reuploading  
  - **Multi-Block Transformers**: Stack attention+FFN blocks for deeper language understanding  
  - **Parameter Shift Gradient** training for quantum gates  
- **GUI Support**  
  - **QelmGUI**: Train/infer on quantum LLMs with real-time logs, progress bars, resource tracking  
  - **QELMChatUI**: Chat-like interface for multi-turn conversations, model selection, and conversation saving  
- **Multi-Threaded / Multiprocessing**  
  - Parallel parameter-shift evaluations  
  - CPU/GPU/both simulation modes  
- **Dataset Flexibility**  
  - Load real text or generate synthetic tokens  
  - Manage token mappings easily  
- **Resource Monitoring**  
  - CPU usage via `psutil`  
  - GPU usage (if available) with `nvidia-smi`

---

## Installation

### Prerequisites
- **Python 3.7+** (up to 3.11 tested)
- **Qiskit** + **Qiskit Aer**
- **TensorFlow**
- **NumPy**
- **Tkinter** (standard in most Python distributions)
- **psutil** *(optional for resource usage)*

### Clone the Repository
```bash
git clone https://github.com/R-D-BioTech-Alaska/QELM.git
cd QELM
```

### Setup Virtual Environment
```bash
python -m venv qiskit_env
# Activate the env:
source qiskit_env/bin/activate     # Linux/macOS
qiskit_env\Scripts\activate        # Windows
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Training the Model
1. **Prepare your dataset**: real text or synthetic (auto-generated).
2. **Set hyperparameters**: vocabulary size, embed dim, #heads, #blocks, advanced ansatz toggles, etc.
3. **Run training**:
   - **GUI**: Launch `QelmGUI.py`, fill in parameters, press **Start Training**.
   - **CLI**: Use `Qelm2.py --train --epochs N --lr 0.05` (older approach).

### Performing Inference
- **GUI**: Inference tab allows user to provide a token, set `max_length`, temperature, and generate.  
- **CLI**: Use `Qelm2.py --inference --input_id 5 --load_path your_model.qelm`.

---

## Graphical Interfaces

### 1. QelmGUI (Training + Inference)
`QelmGUI.py` offers:
- **Dataset Selection** (real .txt or synthetic)
- **Hyperparameter Entry** (embed dim, #heads, #blocks, advanced ansatz, etc.)
- **Live Logs & Progress Bars** (epoch progress, gradient progress)
- **Error & Resource Monitoring** (CPU%, GPU usage if available)
- **Model Save/Load** + **Token Mapping** management
- **Inference** interface (token-based text generation)

**Run**:
```bash
python QelmGUI.py
```
You’ll see a tabbed window for training, inference, and token mapping. Advanced toggles let you experiment with ring entanglement, RZ gates, data reuploading, multi-block architectures, etc.

### 2. QELMChatUI (Conversational UI)
`QELMChatUI.py` provides a **ChatGPT-like** experience:
- **Multi-session**: Keep track of multiple conversation threads
- **Center Chat Panel**: Type messages, get QELM’s replies
- **Advanced Layout**: Avoids duplication errors from older prototypes
- **Model Loading & Token Mapping**: Quickly switch or update quantum LLMs
- **Save Chat**: Archive entire dialogues to text

**Run**:
```bash
python QELMChatUI.py
```
Engage in interactive conversation with your quantum model. Great for testing QELM’s dialogue capabilities or showcasing quantum-based reasoning in a chat interface.

---

## Legacy Command Line (Older Script)
We retain the **original** CLI script `Qelm2.py` for those who want a simpler, command-line-driven approach:
- **Training** (`--train`)  
- **Inference** (`--inference`)  
- Basic model load/save  

However, it lacks the robust features of the GUIs. For a more comprehensive experience, use **QelmGUI**.

---

### Viewing Help/Usage
- **GUI** usage: intuitive once launched; each tab explains itself.
- **CLI** usage:
  ```bash
  python Qelm2.py --help
  ```

---

## Project Structure
```plaintext
QELM/
├── Qelm2.py                # Legacy CLI script for training & inference
├── QelmGUI.py              # Graphical interface for training & inference
├── QELMChatUI.py           # Chat-style interface (like ChatGPT)
├── requirements.txt        # Dependencies
├── README.md               # This documentation
└── docs/
    └── images/
        ├── QELM_Diagram.png
        ├── quantum.png
        └── Qelm.png
```

---

## Credits
If you build upon QELM, please acknowledge:

- **"Based on Quantum-Enhanced Language Model (QELM) by Brenton Carter (Inserian)"**  

- [R-D-BioTech-Alaska/QELM](https://github.com/R-D-BioTech-Alaska/QELM)  

- [Qiskit](https://qiskit.org) community & [IBM Quantum](https://www.ibm.com/quantum)  

---

## License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact
For questions, suggestions, or collaborations:

- **Email**: [contact@rdbiotechalaska.com](mailto:contact@rdbiotechalaska.com)  

- **GitHub**: [R-D-BioTech-Alaska](https://github.com/R-D-BioTech-Alaska)  

- **Website**: [RDBioTech.org](http://RDBioTech.org) or [Qelm.net](http://www.Qelm.net)

> *Disclaimer: QELM is mostly experimental. Community feedback & contributions are welcome and needed to advance this exciting field.*  
