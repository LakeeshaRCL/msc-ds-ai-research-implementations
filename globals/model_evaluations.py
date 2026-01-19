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
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    return acc

def evaluate_precision(y_true, y_pred, average='weighted'):
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    print(f"Precision ({average}): {precision:.4f}")
    return precision

def evaluate_recall(y_true, y_pred, average='weighted'):
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    print(f"Recall ({average}): {rec:.4f}")
    return rec

def evaluate_f1_score(y_true, y_pred, average='weighted'):
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    print(f"F1-Score ({average}): {f1:.4f}")
    return f1

def evaluate_roc_auc(y_true, y_pred_proba, multi_class='ovr', plot=False):
    try:
        # ensure y_true is 1d
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # binary classification
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
            # multi-class
            auc = roc_auc_score(y_true, y_pred_proba, multi_class=multi_class, average='weighted')
            print(f"ROC AUC Score ({multi_class}, weighted): {auc:.4f}")

        return auc

    except Exception as e:
        print(f"Error calculating ROC AUC: {str(e)}")
        return None

def evaluate_computational_performance(model, X_train, y_train, X_test, model_type='sklearn', n_runs=5):
    print("\n=== Computational Performance ===")
    
    train_times = []

    if model_type == 'keras':
        # keras models
        for i in range(n_runs):
            model_config = model.get_config()
            from tensorflow import keras
            temp_model = keras.Sequential.from_config(model_config)
            temp_model.compile(
                optimizer=model.optimizer.__class__.__name__.lower(),
                loss=model.loss,
                metrics=['accuracy']
            )

            start = time.time()
            temp_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
            end = time.time()
            train_times.append(end - start)
    else:
        # sklearn/xgboost models
        for i in range(n_runs):
            if model_type == 'xgboost':
                temp_model = model.__class__(**model.get_params())
            else:
                from sklearn.base import clone
                temp_model = clone(model)

            start = time.time()
            temp_model.fit(X_train, y_train)
            end = time.time()
            train_times.append(end - start)

        model = temp_model

    avg_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)

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

def comprehensive_evaluation(model, X_train, y_train, X_test, y_test, model_type='sklearn', average='weighted', plot_roc=False, epochs=10, batch_size=32):
    print("=" * 60)
    print(f"COMPREHENSIVE MODEL EVALUATION - {model_type.upper()}")
    print("=" * 60)

    if model_type == 'keras':
        y_pred_proba = model.predict(X_test, verbose=0)
        
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test.flatten()

        has_proba = True
    else:
        y_pred = model.predict(X_test)
        y_test_labels = y_test

        try:
            y_pred_proba = model.predict_proba(X_test)
            has_proba = True
        except AttributeError:
            print("Warning: predict_proba() not supported. skipping roc auc.")
            has_proba = False

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
    roc_auc = evaluate_roc_auc(y_test_labels, y_pred_proba, plot=plot_roc) if has_proba else None

    print("\n6. COMPUTATIONAL PERFORMANCE")
    print("-" * 40)
    perf_metrics = evaluate_computational_performance(model, X_train, y_train, X_test, model_type=model_type)

    print("\n" + "=" * 60)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'computational_performance': perf_metrics
    }

def classification_metrics(y_true, y_prob, threshold=0.5):
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
    """find threshold that maximizes f1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    f1_scores = np.nan_to_num(f1_scores)
    
    # ignore last precision/recall value (1.0, 0.0) which has no threshold
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1, {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(best_f1)
    }
