import time
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from globals.pandas_functions import *


class SHAPExplainer:
    """
    Universal SHAP Explainer for SVM, XGBoost, MLP, and Autoencoder
    """

    VALID_MODEL_TYPES = ['svm', 'xgboost', 'mlp', 'autoencoder']

    def __init__(self, model, X_train, X_test, feature_names,
                 model_type, sample_size=100):
        """
        Initialize SHAP Explainer

        Parameters:
        -----------
        model : object
            Trained model (SVM, XGBoost, MLP, or Autoencoder)
        X_train : pd.DataFrame or np.ndarray
            Training data
        X_test : pd.DataFrame or np.ndarray
            Test data
        feature_names : list
            List of feature names
        model_type : str
            One of: 'svm', 'xgboost', 'mlp', 'autoencoder'
        sample_size : int
            Background sample size for KernelExplainer (default: 100)
        """
        # Validate model_type
        if model_type.lower() not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of: {', '.join(self.VALID_MODEL_TYPES)}"
            )

        self.model = model
        self.X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
        self.X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        self.feature_names = feature_names
        self.model_type = model_type.lower()
        self.sample_size = sample_size
        self.shap_explainer = None
        self.shap_values = None
        self.test_samples = None

        # Set explainer type based on model_type
        self.explainer_type = self._get_explainer_type()

        print(f"Initialized SHAPExplainer:")
        print(f"  Model type: {self.model_type}")
        print(f"  Explainer type: {self.explainer_type}")
        print(f"  Features: {len(feature_names)}")

    def _get_explainer_type(self):
        """Map model_type to explainer type"""
        mapping = {
            'svm': 'kernel',
            'xgboost': 'tree',
            'mlp': 'deep',
            'autoencoder': 'deep'
        }
        return mapping[self.model_type]

    def _get_predict_function(self):
        """Get appropriate prediction function based on model type"""
        if self.model_type == 'autoencoder':
            # For autoencoder, use reconstruction error as anomaly score
            return self._autoencoder_predict_proba
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            return self.model.predict
        else:
            raise ValueError("Model must have predict_proba or predict method")

    def _autoencoder_predict_proba(self, X):
        """
        Convert autoencoder reconstruction error to fraud probability
        Higher reconstruction error = higher fraud probability
        """
        # Get reconstructions
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        reconstructions = self.model.predict(X_array)

        # Calculate reconstruction error (MSE per sample)
        mse = np.mean(np.square(X_array - reconstructions), axis=1)

        # Normalize to [0, 1] range using training data statistics
        if not hasattr(self, '_mse_threshold'):
            # Calculate threshold from training data (cached)
            train_reconstructions = self.model.predict(self.X_train.values[:1000])
            train_mse = np.mean(np.square(self.X_train.values[:1000] - train_reconstructions), axis=1)
            self._mse_threshold = np.percentile(train_mse, 95)  # 95th percentile as max

        # Normalize: higher MSE = higher fraud probability
        prob_fraud = np.clip(mse / self._mse_threshold, 0, 1)
        prob_legit = 1 - prob_fraud

        return np.column_stack([prob_legit, prob_fraud])

    def explain(self, test_samples_count=20):
        """
        Compute SHAP values using the appropriate explainer

        Parameters:
        -----------
        test_samples_count : int
            Number of test samples to explain (default: 20)
        """
        # Initialize JS visualization code
        shap.initjs()

        print(f"\n{'=' * 70}")
        print(f"Computing SHAP Explanations for {self.model_type.upper()}")
        print(f"Using {self.explainer_type.upper()} Explainer")
        print(f"{'=' * 70}")

        start_time = time.time()

        # Create explainer based on model type
        if self.model_type == 'xgboost':
            # TreeExplainer for XGBoost (10-100x faster!)
            print("Creating TreeExplainer...")
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("  ✓ TreeExplainer provides exact SHAP values for tree models")

        elif self.model_type in ['mlp', 'autoencoder']:
            # DeepExplainer for neural networks
            print("Creating DeepExplainer...")
            background_size = min(100, len(self.X_train))
            background_data = self.X_train.values[:background_size]
            self.shap_explainer = shap.DeepExplainer(self.model, background_data)
            print(f"  ✓ Using {background_size} background samples")

        elif self.model_type == 'svm':
            # KernelExplainer for SVM (model-agnostic)
            print("Creating KernelExplainer...")
            background_data = shap.sample(self.X_train, self.sample_size)
            predict_fn = self._get_predict_function()
            self.shap_explainer = shap.KernelExplainer(predict_fn, background_data)
            print(f"  ✓ Using {self.sample_size} background samples")
            print("  ⚠ This is slower but works with any model")

        shap_init_time = time.time() - start_time
        print(f"\n✓ Explainer initialization time: {shap_init_time:.2f} seconds")

        # Select test samples to explain
        self.test_samples = self.X_test.iloc[:test_samples_count, :]
        print(f"\nCalculating SHAP values for {test_samples_count} test samples...")

        start_time = time.time()

        # Compute SHAP values
        if self.model_type in ['mlp', 'autoencoder']:
            # DeepExplainer expects numpy arrays
            self.shap_values = self.shap_explainer.shap_values(self.test_samples.values)
        else:
            self.shap_values = self.shap_explainer.shap_values(self.test_samples)

        shap_compute_time = time.time() - start_time
        print(f"✓ SHAP values computation time: {shap_compute_time:.2f} seconds")

        # Display statistics
        if isinstance(self.shap_values, list):
            print(f"  SHAP values shape: {len(self.shap_values)} classes")
            print(f"    Class 0 (Legitimate): {self.shap_values[0].shape}")
            print(f"    Class 1 (Fraud): {self.shap_values[1].shape}")
        else:
            print(f"  SHAP values shape: {self.shap_values.shape}")

        print(f"{'=' * 70}\n")

    def get_shap_values(self):
        """Get computed SHAP values"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")
        return self.shap_values

    def get_explainer(self):
        """Get SHAP explainer object"""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized yet. Call explain() first.")
        return self.shap_explainer

    def show_global_summary_plots(self):
        """
        Display global feature importance plots
        Shows which features are most important across all predictions
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call explain() first.")

        # Extract fraud class SHAP values (class 1)
        if isinstance(self.shap_values, list):
            shap_vals_to_plot = self.shap_values[1]
        elif len(self.shap_values.shape) == 3:
            shap_vals_to_plot = self.shap_values[:, :, 1]
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
            base_value = self.shap_explainer.expected_value[1]
            shap_vals = self.shap_values[1][instance_idx, :]
        elif len(self.shap_values.shape) == 3:
            # Multi-dimensional array (samples, features, classes)
            if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)):
                base_value = self.shap_explainer.expected_value[1]
            else:
                base_value = self.shap_explainer.expected_value
            shap_vals = self.shap_values[instance_idx, :, 1]
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
            shap_vals = self.shap_values[1]
        elif len(self.shap_values.shape) == 3:
            shap_vals = self.shap_values[:, :, 1]
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