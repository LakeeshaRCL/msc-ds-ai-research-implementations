# An Explainable Deep Learning Model with Nature-Inspired Optimization for Fraud Detection in FinTech

**Author:** Lakeesha Ramanayaka  
**Index Number:** MSC/DSA/134  
**University:** University of Sri Jayewardenepura, Sri Lanka  
**Faculty:** Faculty of Graduate Studies

This repository contains the Python implementation for the research project, utilizing Deep Learning models (MLP, Autoencoders) and Nature-Inspired Optimization algorithms to detect fraud in FinTech environments.

## Components

The codebase is structured into modular components:

- **`globals/`**: Core logic and shared modules.
  - `model_evaluations.py`: Comprehensive metrics for model performance (Accuracy, F1, ROC-AUC).
  - `torch_gpu_processing.py`: PyTorch-based model building, training, and GPU acceleration logic (supports CUDA and DirectML).
  - `hyperparameter_optimizer.py`: Integration with Mealpy for swarm-based hyperparameter optimization (PSO, GWO, FA).
  - `data_visualizations.py`: Utilities for plotting and data inspection.

- **`data/`**: Data management.
  - Preprocessing scripts and test datasets (IEEE-CIS Fraud Detection).

- **`utils/`**: Helper utilities.
  - Validation helpers and common tools.

## Requirements

To set up the environment, install the required packages:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Python 3.8+
- **PyTorch**: Deep learning framework.
- **Torch-DirectML**: GPU acceleration support for DirectML devices.
- **Mealpy**: Evolutionary optimization algorithms.
- **Scikit-learn**: traditional machine learning metrics and utilities.
- **Pandas & NumPy**: Data manipulation.

## Usage

The implementation is primarily designed to be run via the provided Jupyter Notebooks, which leverage the modules in `globals` for streamlined execution.
