import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
import globals.hyperparmeter_optimizer as hyp_optimizer
from sklearn.model_selection import StratifiedKFold
import globals.model_evaluations as evaluations
import numpy as np

def configure_gpu(memory_growth=True, enable_xla=True, use_mixed_precision=True):
    if memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs: {gpus}")
        for g in gpus:
            print(f"Configuring memory growth for GPU: {g}")
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    # Optional: enable XLA JIT
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    # Mixed precision speeds up on modern GPUs; keep final Dense in float32
    if use_mixed_precision and tf.config.list_physical_devices('GPU'):
        try:
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            # Policy might have been set already; ignore
            pass

def get_tf_strategy(auto_multi_gpu=True):
    """
    Returns a distribution strategy:
      - MirroredStrategy if multiple GPUs and auto_multi_gpu=True
      - Default (no distribution) otherwise
    """
    if auto_multi_gpu and len(tf.config.list_logical_devices('GPU')) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()

def build_mlp_model(hyperparameters, input_shape,  lr=0.001):

    with get_tf_strategy().scope():
        try:
            policy = mixed_precision.global_policy()
            mixed_active = (policy and policy.name == 'mixed_float16')
        except Exception:
            mixed_active = False

    # building the model

    inputs = tf.keras.Input(shape=(input_shape,), name="inputs")
    x = inputs
    for i, u in enumerate(hyperparameters["units_per_layer"]):
        x = tf.keras.layers.Dense(u, name=f"dense_{i+1}")(x)

        if hyperparameters["batch_norm"]:
           x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)

        x = hyp_optimizer.get_activation_layer(hyperparameters["activation"])(x)

        if hyperparameters["dropout_rate"] > 0:
            x = tf.keras.layers.Dropout(rate=hyperparameters["dropout_rate"], name=f"dropout_{i+1}")(x)

    out_dtype = "float32" if mixed_active else None  # IMPORTANT for mixed precision: keep the output in float32 for stable sigmoid/BCE
    outputs = tf.keras.layers.Dense(1, name="probability", activation="sigmoid", dtype=out_dtype)(x)
    mlp_model = tf.keras.Model(inputs, outputs, name="fraud_detection_mlp_model")
    mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy")

    return mlp_model


def set_optimizer_objective(X, y, cv,  max_epochs, batch_size, seed, early_stopping_patience):
    """
    Build the objective: minimize (1 - mean F1 across CV folds).
    Simplicity choices:
      - Fixed threshold=0.5 for F1
      - EarlyStopping on val_loss
      - No SMOTE here (assume your training set is already prepared)
    """
    splits = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed).split(np.zeros_like(y), y))

    def obj(vec):
        hp = hyp_optimizer.optimizer_vectors_to_mlp_hyperparams( vec)
        f1_scores = []
        for tr_idx, va_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            mlp_model = build_mlp_model(hp, X.shape[1], 1e-3)  # LR 0.001 fixed for simplicity
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True, verbose=0)
            mlp_model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=max_epochs, batch_size=batch_size, verbose=0, callbacks=[es])

            y_prob = mlp_model.predict(X_va, verbose=0).ravel()
            f1_scores.append(evaluations.classification_metrics(y_va, y_prob, threshold=0.5)["f1"])

        # minimize 1 - mean(F1)
        return 1.0 - float(np.mean(f1_scores))

    return obj


def retrain_and_evaluate(best_hp, X_train, y_train, X_test, y_test, batch_size,
                         max_epochs=40, early_stopping_patience=8):
    """
    Retrain the best architecture on all training data (with val_split=0.1),
    then evaluate on the untouched test set at threshold=0.5.
    """
    mlp_model = build_mlp_model(best_hp, X_train.shape[1], 1e-3)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
    mlp_model.fit(X_train, y_train, validation_split=0.1, epochs=max_epochs, batch_size=batch_size, verbose=1, callbacks=[es])

    y_prob = mlp_model.predict(X_test, verbose=0).ravel()
    metrics = evaluations.classification_metrics(y_test, y_prob, threshold=0.5)
    return mlp_model, metrics