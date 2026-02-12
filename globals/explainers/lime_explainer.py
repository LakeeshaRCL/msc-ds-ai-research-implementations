import lime
import lime.lime_tabular
import time
import matplotlib.pyplot as plt

class LimeExplainer:
    def __init__(self, X_train, X_test, feature_names, predict_proba):
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.predict_proba = predict_proba
        self.lime_explainer = None
        self.lime_values = None

    def init_lime_explainer(self):
        # Create LIME Explainer
        print("Generating LIME Explainer...")

        start_time = time.time()

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.to_numpy(),
            feature_names=self.feature_names,
            # class_names=['Non-Fraud', 'Fraud'],
            mode='classification'
        )

        lime_init_time = time.time() - start_time

        print(f"LIME Explainer initialization time: {lime_init_time:.2f} seconds")

    def explain_instance(self, instance_idx, num_features=10):

        # Explain a specific instance
        instance = self.X_test.iloc[instance_idx].values

        print(f"Explaining instance {instance_idx} with LIME...")

        start_time = time.time()
        lime_exp = self.lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=self.predict_proba,
            num_features=num_features
        )

        lime_compute_time = time.time() - start_time
        print(f"LIME explanation computation time: {lime_compute_time:.2f} seconds")
        return lime_exp


    def show_lime_explainer(self, instance_idx, num_features):
        lime_exp = self.explain_instance(instance_idx, num_features)

        # Show LIME explanation in notebook (HTML)
        # lime_exp.show_in_notebook(show_table=True)

        # Display as pyplot figure for static view
        print("LIME Explanation Plot:")
        lime_exp.as_pyplot_figure()
        plt.show()