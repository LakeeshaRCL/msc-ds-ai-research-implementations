import shap
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from globals.torch_gpu_processing import shared_select_device
import warnings
from globals.pandas_functions import *

# Suppress some SHAP warnings that might occur with PyTorch
warnings.filterwarnings("ignore", message=".*The default behavior of `shap.DeepExplainer`.*")

class SHAPExplainerTorch:
    def __init__(self, model, X_train, X_test, feature_names, class_names=['Non-Fraud', 'Fraud']):
        """
        SHAP Explainer for PyTorch Models (DeepExplainer)
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained PyTorch model
        X_train : pd.DataFrame or np.ndarray
            Training data (used for background dataset)
        X_test : pd.DataFrame or np.ndarray
            Test data
        feature_names : list
            List of feature names
        class_names : list
            List of class names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.class_names = class_names
        self.shap_explainer = None
        self.shap_values = None
        self.test_samples = None
        
        self.device = shared_select_device()
        self.model.to(self.device)
        self.model.eval()

    def explain(self, test_samples_count=20, background_size=100):
        """
        Compute SHAP values using DeepExplainer.
        
        Parameters:
        -----------
        test_samples_count : int
            Number of test samples to explain
        background_size : int
            Number of background samples to use (subset of X_train)
        """
        shap.initjs()
        
        print(f"\n{'=' * 70}")
        print(f"Computing SHAP Explanations for PyTorch Model")
        print(f"Using DeepExplainer")
        print(f"{'=' * 70}")
        
        # Prepare background data
        if hasattr(self.X_train, 'values'):
            X_train_np = self.X_train.values
        else:
            X_train_np = self.X_train
            
        # Select background samples
        if len(X_train_np) > background_size:
            background_data = X_train_np[:background_size]
        else:
            background_data = X_train_np
            
        # Convert background to tensor
        background_tensor = torch.tensor(background_data, dtype=torch.float32).to(self.device)
        
        print("Creating DeepExplainer...")
        start_time = time.time()
        
        try:
            # value of DeepExplainer is expected to be a tensor or list of tensors
            self.shap_explainer = shap.DeepExplainer(self.model, background_tensor)
            print(f"    Using {len(background_data)} background samples")
        except Exception as e:
            print(f"Error initializing DeepExplainer: {e}")
            return

        shap_init_time = time.time() - start_time
        print(f"\n    Explainer initialization time: {shap_init_time:.2f} seconds")

        # Select test samples
        if hasattr(self.X_test, 'iloc'):
            self.test_samples = self.X_test.iloc[:test_samples_count, :]
            test_samples_np = self.test_samples.values
        else:
            self.test_samples = pd.DataFrame(self.X_test[:test_samples_count], columns=self.feature_names)
            test_samples_np = self.test_samples.values
            
        test_tensor = torch.tensor(test_samples_np, dtype=torch.float32).to(self.device)
        
        print(f"\nCalculating SHAP values for {test_samples_count} test samples...")
        start_time = time.time()
        
        # Compute SHAP values
        # shap_values from DeepExplainer will be a list of numpy arrays (for each output)
        # or a single numpy array
        try:
            shap_values_raw = self.shap_explainer.shap_values(test_tensor)
            
            # Post-processing to ensure consistent format with original SHAP wrapper
            if isinstance(shap_values_raw, list):
                self.shap_values = shap_values_raw
            else:
                self.shap_values = shap_values_raw
                
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return

        shap_compute_time = time.time() - start_time
        print(f"    SHAP values computation time: {shap_compute_time:.2f} seconds")
        
        # Display statistics
        if isinstance(self.shap_values, list):
            print(f"  SHAP values shape: {len(self.shap_values)} classes")
            for i, sv in enumerate(self.shap_values):
                print(f"    Class {i}: {sv.shape}")
        else:
            print(f"  SHAP values shape: {self.shap_values.shape}")
            
        print(f"{'=' * 70}\n")
        
    def show_summary_plot(self):
        """Display global feature importance summary plot."""
        if self.shap_values is None:
            print("No SHAP values computed.")
            return

        # Prepare data for plotting
        # If output is single array (Regression or Binary Sigmoid), stick with it.
        # If list, usually index 1 is positive class for binary.
        
        if isinstance(self.shap_values, list):
            # for binary classification, usually interested in class 1
            if len(self.shap_values) > 1:
                shap_vals_to_plot = self.shap_values[1]
            else:
                shap_vals_to_plot = self.shap_values[0]
        else:
            shap_vals_to_plot = self.shap_values
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_to_plot, self.test_samples, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot', fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_force_plot(self, instance_idx=0):
        """Display force plot for a single instance."""
        if self.shap_values is None:
            return

        # Handle different structures
        if isinstance(self.shap_values, list):
             # Binary: expected_value is also a list usually, or single value depending on version
            if isinstance(self.shap_explainer.expected_value, list):
                base_value = self.shap_explainer.expected_value[1]
            else:
                base_value = self.shap_explainer.expected_value
                
            shap_values = self.shap_values[1][instance_idx]
        else:
            if isinstance(self.shap_explainer.expected_value, list):
                 base_value = self.shap_explainer.expected_value[0]
            else:
                base_value = self.shap_explainer.expected_value
            shap_values = self.shap_values[instance_idx]

        shap.force_plot(
            base_value,
            shap_values,
            self.test_samples.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=True
        )

    def show_global_summary_plots(self):
        """
        Display global feature importance plots
        Shows which features are most important across all predictions
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        # Extract fraud class SHAP values (class 1 for softmax, class 0 for sigmoid)
        if isinstance(self.shap_values, list):
            shap_vals_to_plot = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        elif len(self.shap_values.shape) == 3:
            class_idx = 1 if self.shap_values.shape[2] > 1 else 0
            shap_vals_to_plot = self.shap_values[:, :, class_idx]
        else:
            shap_vals_to_plot = self.shap_values

        # Bar plot - shows mean absolute SHAP values
        print("Generating SHAP Summary Plot (Bar)...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals_to_plot,
            self.test_samples,
            plot_type="bar",
            feature_names=self.feature_names,
            show=False
        )
        plt.title('Global Feature Importance - Mean |SHAP Value|', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Dot plot - shows feature importance with value distribution
        print("Generating SHAP Summary Plot (Dot)...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals_to_plot,
            self.test_samples,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('Feature Importance with Value Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def get_instance_and_base_shap_values(self, instance_idx):
        """
        Get SHAP values and base value for a specific instance

        Parameters:
        -----------
        instance_idx : int
            Index of the instance to explain

        Returns:
        --------
        base_value : float
            Expected value (baseline prediction)
        shap_vals : np.ndarray
            SHAP values for the instance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        print(f"\nExtracting SHAP values for instance {instance_idx}:")

        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # Binary classification with list format
            if len(self.shap_values) > 1:
                base_value = self.shap_explainer.expected_value[1]
                shap_vals = self.shap_values[1][instance_idx, :]
            else:
                base_value = self.shap_explainer.expected_value[0] if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) else self.shap_explainer.expected_value
                shap_vals = self.shap_values[0][instance_idx, :]
        elif len(self.shap_values.shape) == 3:
            # Multi-dimensional array (samples, features, classes)
            class_idx = 1 if self.shap_values.shape[2] > 1 else 0
            if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)):
                ev = self.shap_explainer.expected_value
                base_value = ev[class_idx] if len(ev) > class_idx else ev[0]
            else:
                base_value = self.shap_explainer.expected_value
            shap_vals = self.shap_values[instance_idx, :, class_idx]
        else:
            # 2D array format
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            shap_vals = self.shap_values[instance_idx, :]

        print(f"  Base value (expected): {base_value:.4f}")
        print(f"  SHAP values shape: {shap_vals.shape}")
        print(f"  Sum of SHAP values: {np.sum(shap_vals):.4f}")
        print(f"  Prediction: {base_value + np.sum(shap_vals):.4f}")

        return base_value, shap_vals

    def show_waterfall_plot(self, instance_idx):
        """
        Display waterfall plot for a specific instance
        Shows cumulative contribution of each feature

        Parameters:
        -----------
        instance_idx : int
            Index of the instance to explain
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        print(f"\n{'=' * 70}")
        print(f"SHAP Waterfall Plot for Instance {instance_idx}")
        print(f"{'=' * 70}")

        base_value, shap_vals = self.get_instance_and_base_shap_values(instance_idx)

        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=self.test_samples.iloc[instance_idx].values,
            feature_names=self.feature_names
        )

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'Waterfall Plot - Instance {instance_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def get_feature_influence(self, instance_idx, export_csv_path=None, export_file_name=None):
        """
        Get detailed feature influences for a specific instance

        Parameters:
        -----------
        instance_idx : int
            Index of the instance to analyze
        export_csv_path : str, optional
            Path to export CSV
        export_file_name : str, optional
            Name of the CSV file

        Returns:
        --------
        feature_influence_df : pd.DataFrame
            DataFrame with feature influences sorted by importance
        """
        print(f"\n{'=' * 70}")
        print(f"Detailed Feature Influences for Instance {instance_idx}")
        print(f"{'=' * 70}")

        # Get instance data
        instance_data = self.test_samples.iloc[instance_idx]

        # Get SHAP values
        base_value, instance_shap_values = self.get_instance_and_base_shap_values(instance_idx)

        # Create DataFrame
        feature_influence_df = pd.DataFrame({
            'Dataset Feature Index': self.test_samples.columns,
            'Feature Name': self.feature_names,
            'Feature Value': instance_data.values,
            'SHAP Value (Influence)': instance_shap_values
        })

        # Sort by absolute influence
        feature_influence_df['Abs Influence'] = feature_influence_df['SHAP Value (Influence)'].abs()
        feature_influence_df = feature_influence_df.sort_values(
            by='Abs Influence',
            ascending=False
        ).drop(columns=['Abs Influence'])

        # Display top 15 features
        print("\nTop 15 Most Influential Features:")
        print(feature_influence_df.head(15).to_string(index=False))

        # Export if requested
        if export_csv_path is not None and export_file_name is not None:
            export_dataframe_to_csv(feature_influence_df, export_csv_path, export_file_name)
            print(f"\n✓ Exported to: {export_csv_path}/{export_file_name}")

        return feature_influence_df

    def get_top_features_summary(self, top_n=10):
        """
        Get summary of top N most important features across all samples

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        top_features_df : pd.DataFrame
            DataFrame with mean absolute SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        # Extract fraud class SHAP values
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        elif len(self.shap_values.shape) == 3:
            class_idx = 1 if self.shap_values.shape[2] > 1 else 0
            shap_vals = self.shap_values[:, :, class_idx]
        else:
            shap_vals = self.shap_values

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        # Create DataFrame
        top_features_df = pd.DataFrame({
            'Feature Name': self.feature_names,
            'Mean |SHAP Value|': mean_abs_shap
        }).sort_values(by='Mean |SHAP Value|', ascending=False)

        print(f"\nTop {top_n} Most Important Features (Global):")
        print(top_features_df.head(top_n).to_string(index=False))

        return top_features_df.head(top_n)

    def compare_instances(self, instance_indices, top_n=10):
        """
        Compare SHAP explanations across multiple instances

        Parameters:
        -----------
        instance_indices : list
            List of instance indices to compare
        top_n : int
            Number of top features to show per instance
        """
        print(f"\n{'=' * 70}")
        print(f"Comparing {len(instance_indices)} Instances")
        print(f"{'=' * 70}")

        for idx in instance_indices:
            print(f"\n--- Instance {idx} ---")
            self.get_feature_influence(idx)
            print()

    def show_local_interpretability(self, instance_idx):
        """
        Display force plot for a specific instance
        Shows how each feature contributes to the prediction

        Parameters:
        -----------
        instance_idx : int
            Index of the instance to explain
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        print(f"\n{'=' * 70}")
        print(f"SHAP Force Plot for Instance {instance_idx}")
        print(f"{'=' * 70}")

        base_value, shap_vals = self.get_instance_and_base_shap_values(instance_idx)

        # Create force plot
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            base_value,
            shap_vals,
            self.test_samples.iloc[instance_idx, :],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'Force Plot - Instance {instance_idx}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

