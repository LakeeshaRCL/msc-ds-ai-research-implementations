import lime
import lime.lime_tabular
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from globals.torch_gpu_processing import shared_select_device

class LimeExplainerTorch:
    def __init__(self, model, X_train, X_test, feature_names, class_names=['Non-Fraud', 'Fraud']):
        """
        LIME Explainer for PyTorch Models
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained PyTorch model
        X_train : pd.DataFrame or np.ndarray
            Training data (used for initialization statistics)
        X_test : pd.DataFrame or np.ndarray
            Test data
        feature_names : list
            List of feature names
        class_names : list
            List of class names (default: ['Non-Fraud', 'Fraud'])
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.class_names = class_names
        self.lime_explainer = None
        self.device = shared_select_device()
        
        # Ensure model is in eval mode and on correct device
        self.model.to(self.device)
        self.model.eval()

    def _predict_proba(self, X_numpy):
        """
        Internal prediction wrapper for LIME.
        Converts numpy array to tensor, runs inference, and formats output.
        """
        # Convert to tensor
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # If model creates logits/probabilities
            # Assuming binary classification where output is probability of class 1 (Fraud)
            # or logits.
            
            # Check shape to determine if it's single output (sigmoid) or 2 outputs (softmax)
            if outputs.shape[1] == 1:
                # Single output (probability of class 1)
                probs_1 = outputs.cpu().numpy().flatten()
                probs_0 = 1.0 - probs_1
                return np.column_stack([probs_0, probs_1])
            else:
                # Multiple outputs (softmax logic usually)
                # If logits, apply softmax? The model in notebook ends with Sigmoid() and 1 output unit
                # but let's be safe.
                # Actually, mlp_build_model ends with Sigmoid() -> probabilities.
                return outputs.cpu().numpy()

    def init_lime_explainer(self):
        # Create LIME Explainer
        print("Generating LIME Explainer (PyTorch)...")

        start_time = time.time()
        
        # Convert X_train to numpy if it's a DataFrame
        training_data = self.X_train.to_numpy() if hasattr(self.X_train, 'to_numpy') else self.X_train

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        lime_init_time = time.time() - start_time

        print(f"LIME Explainer initialization time: {lime_init_time:.2f} seconds")

    def explain_instance(self, instance_idx, num_features=10):

        # Explain a specific instance
        if hasattr(self.X_test, 'iloc'):
            instance = self.X_test.iloc[instance_idx].values
        else:
            instance = self.X_test[instance_idx]

        print(f"Explaining instance {instance_idx} with LIME...")

        start_time = time.time()
        lime_exp = self.lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=self._predict_proba,
            num_features=num_features
        )

        lime_compute_time = time.time() - start_time
        print(f"LIME explanation computation time: {lime_compute_time:.2f} seconds")
        return lime_exp


    def show_lime_explainer(self, instance_idx, num_features=10):
        lime_exp = self.explain_instance(instance_idx, num_features)

        # Display as pyplot figure for static view
        print("LIME Explanation Plot:")
        try:
            fig = lime_exp.as_pyplot_figure()
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            
        return lime_exp
