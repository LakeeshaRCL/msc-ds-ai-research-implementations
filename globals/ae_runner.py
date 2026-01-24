
import sys
import os
import torch
import numpy as np
import pandas as pd
from mealpy.swarm_based import FA, GWO, PSO
from mealpy.utils.space import IntegerVar, FloatVar, BoolVar, CategoricalVar

import globals.hyperparameter_optimizer as hyp_optimizer
import globals.torch_gpu_processing as torch_gpu_processing
from globals.pandas_functions import *

def run_ae_initial_setup(data_path):
    print(f"Loading data from: {data_path}")
    X_train = pd.read_csv(f"{data_path}/unified_transaction_data_option2_x_train_scaled.csv")
    y_train = pd.read_csv(f"{data_path}/unified_transaction_data_option2_y_train.csv")
    X_validation = pd.read_csv(f"{data_path}/unified_transaction_data_option2_x_validation_scaled.csv")
    y_validation = pd.read_csv(f"{data_path}/unified_transaction_data_option2_y_validation.csv")
    X_test = pd.read_csv(f"{data_path}/unified_transaction_data_option2_x_test_scaled.csv")
    y_test = pd.read_csv(f"{data_path}/unified_transaction_data_option2_y_test.csv")
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def get_sampling_data(X_train, y_train, sample_size=60000, seed=42):
    y_train_flat = y_train.to_numpy().ravel()
    X_train_np = X_train.to_numpy()

    # Filter Non-Fraud
    mask = (y_train_flat == 0)
    X_clean = X_train_np[mask]
    y_clean = y_train_flat[mask]

    if len(X_clean) > sample_size:
        np.random.seed(seed)
        indices = np.random.choice(len(X_clean), sample_size, replace=False)
        return X_clean[indices], y_clean[indices]

    return X_clean, y_clean

def run_optimization(X_train_sample, y_train_sample, X_validation, y_validation, 
                    algorithm="PSO", population=6, iterations=5, 
                    batch_size=1024, epochs=10, early_stopping=3):
    
    print(f"Starting AE Optimization using {algorithm}...")
    print(f"Settings: Pop={population}, Iter={iterations}, Batch={batch_size}, Epochs={epochs}")

    # filtering validation data to keep only non-fraud for training
    y_val_flat = y_validation.to_numpy().ravel()
    X_val_np = X_validation.to_numpy()
    mask_val_clean = (y_val_flat == 0)
    X_val_clean = X_val_np[mask_val_clean]

    # keep the full validation set to use for AUPRC calculation
    X_val_full = X_val_np
    y_val_full = y_val_flat

    print(f"Starting AE Optimization using {algorithm}...")
    print(f"Training samples (non-fraud): {len(X_train_sample)}")
    print(f"Validation samples (non-fraud for training): {len(X_val_clean)}")
    print(f"Validation samples (full for AUPRC): {len(X_val_full)}")
    
    # objective function to be optimized
    objective_function = torch_gpu_processing.set_ae_optimizer_objective(
        X_train_sample,
        y_train_sample,
        X_val_clean,  # Use this for autoencoder training/validation loss
        X_val_full,  # Use this for computing reconstruction errors
        y_val_full,  # Use this for AUPRC calculation
        max_epochs=epochs,
        batch_size=batch_size,
        seed=42,
        early_stopping_patience=early_stopping
    )

    # define hyperparameter bounds
    bounds_cfg = hyp_optimizer.get_ae_hyperparameter_bounds_config(min_layers=1, max_layers=10)
    optimizer_bounds = []
    for cfg in bounds_cfg:
        if cfg['type'] == 'int':
            optimizer_bounds.append(IntegerVar(lb=cfg['lb'], ub=cfg['ub'], name=cfg.get('name')))
        elif cfg['type'] == 'float':
            optimizer_bounds.append(FloatVar(lb=cfg['lb'], ub=cfg['ub'], name=cfg.get('name')))
        elif cfg['type'] == 'bool':
            optimizer_bounds.append(BoolVar(name=cfg.get('name')))
        elif cfg['type'] == 'categorical':
            optimizer_bounds.append(CategoricalVar(valid_sets=cfg['choices'], name=cfg.get('name')))

    problem = dict(obj_func=objective_function, bounds=optimizer_bounds, minmax="min", log_to=None)

    # select optimizer algorithm
    if algorithm == "PSO":
        optimizer = PSO.OriginalPSO(epoch=iterations, pop_size=population)
    elif algorithm == "GWO":
        optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=population)
    else:
        optimizer = FA.OriginalFA(epoch=iterations, pop_size=population)

    # executing optimization
    best_agent = optimizer.solve(problem)
    best_vec = best_agent.solution
    best_obj = best_agent.target.fitness
    best_hp = hyp_optimizer.optimizer_vectors_to_ae_hyperparams(best_vec)
    
    print(f"\nBest Objective: {best_obj:.6f} => AUPRC: {1.0-best_obj:.6f}")
    
    return best_hp