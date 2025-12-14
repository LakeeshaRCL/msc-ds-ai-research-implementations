"""
model_evaluation.py
Comprehensive model evaluation functions for classification tasks
Compatible with SVM, XGBoost, LSTM, and Autoencoder models
"""
# TODO: need to test this code with actual models to ensure compatibility

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt


def evaluate_accuracy(y_true, y_pred):
    """
    Calculate accuracy score.
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    float : Accuracy score (0 to 1)
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def evaluate_precision(y_true, y_pred, average='weighted'):
    """
    Calculate precision score.
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='weighted'
        Type of averaging ('binary', 'micro', 'macro', 'weighted')

    Returns:
    --------
    float : Precision score (0 to 1)
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    print(f"Precision ({average}): {precision:.4f}")
    return precision


def evaluate_recall(y_true, y_pred, average='weighted'):
    """
    Calculate recall score (sensitivity).
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='weighted'
        Type of averaging ('binary', 'micro', 'macro', 'weighted')

    Returns:
    --------
    float : Recall score (0 to 1)
    """
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    print(f"Recall ({average}): {recall:.4f}")
    return recall


def evaluate_f1_score(y_true, y_pred, average='weighted'):
    """
    Calculate F1 score (harmonic mean of precision and recall).
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='weighted'
        Type of averaging ('binary', 'micro', 'macro', 'weighted')

    Returns:
    --------
    float : F1 score (0 to 1)
    """
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    print(f"F1-Score ({average}): {f1:.4f}")
    return f1


def evaluate_roc_auc(y_true, y_pred_proba, multi_class='ovr', plot=False):
    """
    Calculate Area Under ROC Curve.
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
        - For sklearn models: Use model.predict_proba()
        - For XGBoost: Use model.predict_proba()
        - For LSTM/AE: Use model.predict() output (softmax probabilities)
    multi_class : str, default='ovr'
        Strategy for multi-class ('ovr' or 'ovo')
    plot : bool, default=False
        Whether to plot ROC curve (works for binary classification)

    Returns:
    --------
    float : ROC AUC score (0 to 1)
    """
    try:
        # Ensure y_true is 1D
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        # Check if binary or multi-class
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            # Handle both 2D probability arrays and 1D arrays
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                proba = y_pred_proba[:, 1]
            elif len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 1:
                proba = y_pred_proba[:, 0]
            else:
                proba = y_pred_proba

            auc = roc_auc_score(y_true, proba)
            print(f"ROC AUC Score: {auc:.4f}")

            if plot:
                fpr, tpr, thresholds = roc_curve(y_true, proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.show()
        else:
            # Multi-class classification
            auc = roc_auc_score(y_true, y_pred_proba, multi_class=multi_class, average='weighted')
            print(f"ROC AUC Score ({multi_class}, weighted): {auc:.4f}")

        return auc

    except Exception as e:
        print(f"Error calculating ROC AUC: {str(e)}")
        return None


def evaluate_computational_performance(model, X_train, y_train, X_test,
                                       model_type='sklearn', n_runs=5):
    """
    Evaluate computational performance metrics.
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    model : model object
        Machine learning model (untrained or cloned)
        - SVM: sklearn SVC model
        - XGBoost: XGBClassifier model
        - LSTM: Keras/TensorFlow Sequential model
        - AE: Keras/TensorFlow autoencoder model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels (for LSTM, should be one-hot encoded if needed)
    X_test : array-like
        Testing features
    model_type : str, default='sklearn'
        Type of model: 'sklearn', 'xgboost', 'keras' (for LSTM/AE)
    n_runs : int, default=5
        Number of runs to average timing

    Returns:
    --------
    dict : Dictionary containing timing metrics
    """
    print("\n=== Computational Performance ===")

    # Training time
    train_times = []

    if model_type == 'keras':
        # For Keras models (LSTM/AE)
        for i in range(n_runs):
            # Clone model architecture
            model_config = model.get_config()
            from tensorflow import keras
            temp_model = keras.Sequential.from_config(model_config)
            temp_model.compile(
                optimizer=model.optimizer.__class__.__name__.lower(),
                loss=model.loss,
                metrics=['accuracy']
            )

            start = time.time()
            temp_model.fit(X_train, y_train, epochs=10, batch_size=32,
                           verbose=0, validation_split=0.1)
            end = time.time()
            train_times.append(end - start)
    else:
        # For sklearn and XGBoost models
        for i in range(n_runs):
            # Clone model
            if model_type == 'xgboost':
                temp_model = model.__class__(**model.get_params())
            else:
                from sklearn.base import clone
                temp_model = clone(model)

            start = time.time()
            temp_model.fit(X_train, y_train)
            end = time.time()
            train_times.append(end - start)

        model = temp_model  # Use the last trained model for prediction timing

    avg_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)

    # Prediction time
    pred_times = []
    for i in range(n_runs):
        start = time.time()
        if model_type == 'keras':
            model.predict(X_test, verbose=0)
        else:
            model.predict(X_test)
        end = time.time()
        pred_times.append(end - start)

    avg_pred_time = np.mean(pred_times)
    std_pred_time = np.std(pred_times)

    # Prediction time per sample
    pred_time_per_sample = avg_pred_time / len(X_test)

    print(f"Average Training Time: {avg_train_time:.4f} ± {std_train_time:.4f} seconds")
    print(f"Average Prediction Time: {avg_pred_time:.6f} ± {std_pred_time:.6f} seconds")
    print(f"Prediction Time per Sample: {pred_time_per_sample:.6f} seconds")
    print(f"Throughput: {1 / pred_time_per_sample:.2f} predictions/second")

    return {
        'avg_train_time': avg_train_time,
        'std_train_time': std_train_time,
        'avg_pred_time': avg_pred_time,
        'std_pred_time': std_pred_time,
        'pred_time_per_sample': pred_time_per_sample,
        'throughput': 1 / pred_time_per_sample
    }


def comprehensive_evaluation(model, X_train, y_train, X_test, y_test,
                             model_type='sklearn', average='weighted',
                             plot_roc=False, epochs=10, batch_size=32):
    """
    Perform comprehensive model evaluation using all metrics.
    Compatible with: SVM, XGBoost, LSTM, AE

    Parameters:
    -----------
    model : model object
        Trained machine learning model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Testing features
    y_test : array-like
        Testing labels
    model_type : str, default='sklearn'
        Type of model: 'sklearn', 'xgboost', 'keras'
    average : str, default='weighted'
        Averaging method for multi-class metrics
    plot_roc : bool, default=False
        Whether to plot ROC curve
    epochs : int, default=10
        Number of epochs for Keras models (computational performance)
    batch_size : int, default=32
        Batch size for Keras models

    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    print("=" * 60)
    print(f"COMPREHENSIVE MODEL EVALUATION - {model_type.upper()}")
    print("=" * 60)

    # Get predictions based on model type
    if model_type == 'keras':
        # For Keras models (LSTM/AE)
        y_pred_proba = model.predict(X_test, verbose=0)

        # Handle different output shapes
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Convert y_test if one-hot encoded
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test.flatten()

        has_proba = True
    else:
        # For sklearn and XGBoost models
        y_pred = model.predict(X_test)
        y_test_labels = y_test

        # Get prediction probabilities for ROC AUC
        try:
            y_pred_proba = model.predict_proba(X_test)
            has_proba = True
        except AttributeError:
            print("Warning: Model does not support predict_proba(). ROC AUC will be skipped.")
            has_proba = False

    # Evaluate all metrics
    print("\n1. ACCURACY")
    print("-" * 40)
    accuracy = evaluate_accuracy(y_test_labels, y_pred)

    print("\n2. PRECISION")
    print("-" * 40)
    precision = evaluate_precision(y_test_labels, y_pred, average=average)

    print("\n3. RECALL")
    print("-" * 40)
    recall = evaluate_recall(y_test_labels, y_pred, average=average)

    print("\n4. F1-SCORE")
    print("-" * 40)
    f1 = evaluate_f1_score(y_test_labels, y_pred, average=average)

    print("\n5. ROC AUC")
    print("-" * 40)
    if has_proba:
        roc_auc = evaluate_roc_auc(y_test_labels, y_pred_proba, plot=plot_roc)
    else:
        roc_auc = None

    print("\n6. COMPUTATIONAL PERFORMANCE")
    print("-" * 40)
    perf_metrics = evaluate_computational_performance(
        model, X_train, y_train, X_test, model_type=model_type
    )

    print("\n" + "=" * 60)

    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'computational_performance': perf_metrics
    }

    return results


def classification_metrics(y_true, y_prob, threshold=0.5):
    """Compute standard metrics for binary classification."""
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal decision threshold that maximizes the F1 score.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities for the positive class
        
    Returns:
    --------
    best_threshold : float
        Threshold that maximizes F1 score
    best_f1 : float
        The maximum F1 score achieved
    metrics : dict
        Dictionary containing precision, recall, and f1 at the best threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for each threshold
    # Note: precision_recall_curve returns precisions and recalls with length = len(thresholds) + 1
    # The last precision and recall values are 1.0 and 0.0 respectively, which don't have a corresponding threshold.
    # We ignore the last value for calculation with thresholds.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # Handle NaN values that might arise from division by zero
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find the index of the maximum F1 score
    # We only look at the first len(thresholds) indices
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    
    return best_threshold, best_f1, {
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1": float(best_f1)
    }
