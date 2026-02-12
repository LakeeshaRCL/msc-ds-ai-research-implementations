import time
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from globals.pandas_functions import *


class SHAPExplainer:
    def __init__(self, predict_proba, X_train, X_test, feature_names, sample_size=100):
        self.predict_proba = predict_proba
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.sample_size = sample_size
        self.shap_explainer = None
        self.shap_values = None
        self.test_samples = None

    def explain(self):

        # Initialize JS visualization code
        shap.initjs()

        # Use a small background dataset for efficiency
        # Sampling 100 instances from the training set as background data
        background_data = shap.sample(self.X_train, self.sample_size)

        # Create KernelExplainer
        print("Generating SHAP Explainer...")
        start_time = time.time()
        self.shap_explainer = shap.KernelExplainer(self.predict_proba, background_data)
        shap_init_time = time.time() - start_time
        print(f"SHAP Explainer initialization time: {shap_init_time:.2f} seconds")

        # Explain a subset of test data (e.g., 20 samples) for demonstration
        test_samples_count = 20
        self.test_samples = self.X_test.iloc[:test_samples_count, :]

        print(f"Calculating SHAP values for {test_samples_count} test samples...")
        start_time = time.time()
        self.shap_values = self.shap_explainer.shap_values(self.test_samples)
        shap_compute_time = time.time() - start_time
        print(f"SHAP values computation time: {shap_compute_time:.2f} seconds")

    def get_shap_values(self):
        if self.shap_values is None:
            raise ValueError("SHAP values have not been computed yet. Call explain() first.")
        return self.shap_values

    def get_explainer(self):
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer has not been initialized yet. Call explain() first.")
        return self.shap_explainer

    def show_global_summary_plots(self):
        if self.shap_values is None:
            raise ValueError("SHAP values have not been computed yet. Call explain() first.")
        # Global interpretability: Summary plot
        print("SHAP Summary Plot (Bar):")
        plt.figure()

        # Select Class 1 (Fraud) for explanation
        if isinstance(self.shap_values, list):
            shap_vals_to_plot = self.shap_values[1]
        elif len(self.shap_values.shape) == 3:
            shap_vals_to_plot = self.shap_values[:, :, 1]
        else:
            shap_vals_to_plot = self.shap_values

        shap.summary_plot(shap_vals_to_plot, self.test_samples, plot_type="bar", feature_names=self.feature_names)
        plt.show()

        print("SHAP Summary Plot (Dot):")
        plt.figure()
        shap.summary_plot(shap_vals_to_plot, self.test_samples, feature_names=self.feature_names)
        plt.show()

    def get_instance_and_base_shap_values(self, instance_idx):
        print(f"Instance: {instance_idx}:")

        # Robust selection for Class 1 (Fraud)
        if isinstance(self.shap_values, list):
            base_value = self.shap_explainer.expected_value[1]
            shap_vals = self.shap_values[1][instance_idx, :]
        elif len(self.shap_values.shape) == 3:
            if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)):
                base_value = self.shap_explainer.expected_value[1]
            else:
                base_value = self.shap_explainer.expected_value
            shap_vals = self.shap_values[instance_idx, :, 1]
        else:
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
            shap_vals = self.shap_values[instance_idx, :]

        print(f"Base value: {base_value}")
        print(f"SHAP values for instance {instance_idx}: {shap_vals}")
        return base_value, shap_vals


    def show_local_interpretability(self, instance_idx):
        if self.shap_values is None:
            raise ValueError("SHAP values have not been computed yet. Call explain() first.")
        # Local interpretability: Force plot for a specific instance

        print(f"SHAP Force Plot for instance: {instance_idx}:")

        base_value, shap_vals = self.get_instance_and_base_shap_values(instance_idx)

        print(f"Base value used: {base_value}")
        shap.force_plot(base_value, shap_vals, self.test_samples.iloc[instance_idx, :], matplotlib=True,
                        feature_names=self.feature_names)
        
    def get_feature_influence(self, instance_idx, export_csv_path=None, export_file_name=None):
        # DETAIL: Top Feature Influences for the Specific Instance
        print(f"\nDetailed Feature Influences for Instance: {instance_idx}")

        # Get original feature values
        instance_data = self.test_samples.iloc[instance_idx]

        # Get SHAP values for this instance (already calculated above as shap_vals)
        base_value, instance_shap_values = self.get_instance_and_base_shap_values(instance_idx)

        # Create a DataFrame for better readability
        feature_influence_df = pd.DataFrame({
            'Dataset Feature Index': self.test_samples.columns,
            'Feature Name': self.feature_names,
            'Feature Value': instance_data.values,
            'SHAP Value (Influence)': instance_shap_values
        })

        # Sort by absolute influence to see top drivers
        feature_influence_df['Abs Influence'] = feature_influence_df['SHAP Value (Influence)'].abs()
        feature_influence_df = feature_influence_df.sort_values(by='Abs Influence', ascending=False).drop(
            columns=['Abs Influence'])

        # Display top 15 features
        print(feature_influence_df.head(15))

        if export_csv_path is not None and export_file_name is not None:
            export_dataframe_to_csv(feature_influence_df, export_csv_path, export_file_name)
        