---

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
  - [Performing Inference](#performing-inference)
  - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
  - [Viewing Help/Usage Information](#viewing-helpusage-information)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)
- [Contact](#contact)

---

## Overview

Welcome to the **Quantum-Enhanced Language Model (QELM)** project; an innovative project that merges the power of quantum computing with natural language processing to create a next-generation language model. QELM creates and reduces LLMs to ultra-compact models using qubits without compromising capabilities, enabling them to run instantly on small devices without data centers.

---

## Features

- **Quantum Parameter Optimization**: Gradient-based optimization via the Parameter Shift Rule.
- **Advanced Quantum Circuits**: Implements entangling gates with multiple layers.
- **Thread-Safe GUI with Training Feedback**: New GUI-based interface for training, inference, and model management.
- **Synthetic and Real Dataset Support**: Train with synthetic datasets for testing or real-world text data.
- **Resource Monitoring**: Integrated system resource usage monitoring (CPU/GPU).
- **Enhanced Logging**: Threaded logging ensures training progress and errors are visible in real-time.

---

## Installation

### Prerequisites

- **Python 3.7 to 3.11**
- **Qiskit**, **Qiskit Aer**, **TensorFlow**, **Numpy**, **Tkinter**
- **psutil** (optional for resource monitoring)

---

### Clone the Repository

```bash
git clone https://github.com/R-D-BioTech-Alaska/QELM.git
cd QELM
```

---

### Setup Virtual Environment

```bash
python -m venv qiskit_env
# Activate the virtual environment
source qiskit_env/bin/activate  # For Linux/macOS
qiskit_env\Scripts\activate     # For Windows
```

---

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Training the Model

QELM allows you to train using **synthetic** or **real datasets**. Use the command-line interface (CLI) for basic runs or the GUI for an enhanced experience.

#### CLI Example:

```bash
python Qelm2.py --train --epochs 02 --lr 0.05
```

#### GUI Model with Threading

To run the **QELM GUI**, execute:

```bash
python QelmGUI.py
```

This launches an intuitive interface for:

- Selecting datasets
- Training with live progress updates
- Running inference
- Managing token mappings

**Key GUI Features**:

- Real-time progress bars for gradient computations and training
- System resource usage (CPU/GPU) display
- Interactive logs for feedback and error monitoring
- Buttons for saving/loading models, stopping training gracefully, or halting immediately.

---

### Performing Inference

Run predictions from the GUI or CLI.

#### CLI Inference Example:

```bash
python Qelm2.py --inference --input_id 5 --load_path quantum_llm_model_enhanced.json
```

#### GUI Inference:

1. Enter an **input token**.
2. Set parameters like `Max Length` and `Temperature`.
3. Click **Run Inference**.

The GUI outputs the generated sequence.

---

## Graphical User Interface (GUI)

The **QELM GUI** offers an easy-to-use tool for all functionalities, including:

- **Training**: Monitor training logs, set hyperparameters, and view progress visually.
- **Inference**: Input tokens and generate outputs interactively.
- **Model Management**: Save and load models with token mappings.
- **Resource Monitoring**: View real-time CPU usage and estimate remaining training time.

**Launching the GUI**:

```bash
python QelmGUI.py
```

---

## Project Structure

```plaintext
QELM/
├── Qelm2.py                          # CLI-based model training and inference
├── QelmGUI.py                        # GUI model with threading
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── quantum_llm_model_enhanced.json   # Default model file
└── docs/
    └── images/
        ├── QELM_Diagram.png
        └── quantum.png
```

---

## Credits

If you use or build upon this project, provide proper credit to the original developer:

- Include the following attribution in your project:  
  **"Based on Quantum-Enhanced Language Model (QELM) by Brenton Carter (Inserian)"**

- Provide a link back to the original repository: [R-D-BioTech-Alaska/QELM](https://github.com/R-D-BioTech-Alaska/QELM)

- Include a mention to the Qiskit community. [Qiskit](https://qiskit.org/)

- [IBM Quantum](https://www.ibm.com/quantum)

---

## License

This project is licensed under the MIT License.

---

## Contact

For inquiries, suggestions, or contributions:

- **Email**: [contact@rdbiotechalaska.com](mailto:contact@rdbiotechalaska.com)
- **GitHub**: [R-D-BioTech-Alaska](https://github.com/R-D-BioTech-Alaska)
- **Website**: [RDBioTech.org](http://RDBioTech.org)

---

> **Disclaimer**: QELM is an experimental project integrating quantum principles with NLP. Contributions are welcome to advance this pioneering effort in quantum-enhanced computing! 

---
