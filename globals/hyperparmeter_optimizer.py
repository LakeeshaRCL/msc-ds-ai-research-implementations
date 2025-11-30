import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

activation_functions = ["relu", "leaky_relu"]
batch_sizes =[256, 512, 1024, 2048]
hidden_layer_unit_choices = list(range(16, 257, 8))

def decode_scalar(raw_value, spec):
    val = raw_value

    val = np.clip(val, spec["low"], spec["high"])

    # round to integer if specified
    if spec.get("round"):
        val = int(round(val))

    # handle multiples for integer types
    if "multiple" in spec:
        val = int(round(val / spec["multiple"]) * spec["multiple"])
        val = int(np.clip(val, spec["low"], spec["high"]))

    # Optional transforms (e.g., log space decoding)
    # transform = spec.get("transform")
    # if transform == "pow10":
    #     # Interprets the clipped float as log10(value)
    #     val = 10 ** val

    return val

# Mealpy optimizer will output real-valued vectors. This function maps those vectors to MLP hyperparameters.
def optimizer_vectors_to_mlp_hyperparams(vector):
    """
        Decode the optimizer's continuous vector into a simple MLP architecture.
        Vector layout (10 dims total):
          [0] n_hidden_layers        : int in [1..4]
          [1] units_layer1           : int in [16..256] (multiple of 8)
          [2] units_layer2           : int in [16..256] (multiple of 8)
          [3] units_layer3           : int in [16..256] (multiple of 8)
          [4] units_layer4           : int in [16..256] (multiple of 8)
          [5] dropout_rate           : float in [0.0..0.5]
          [6] activation_code        : int in [0..2] mapping to ACTIVATIONS
          [7] batch_norm_flag        : int in {0,1} -> False/True
          [8] (reserved)             : not used (left for future)
          [9] (reserved)             : not used (left for future)
        Only the first 'n_hidden_layers' unit entries are used.
        """
    # Number of hidden layers
    n_layers = int(np.clip(np.round(vector[0]), 1, 4))

    # Units per layer (rounded to multiples of 8 for efficiency)
    raw_units = vector[1:5]
    units = [int(np.clip(np.round(u / 8) * 8, 16, 256)) for u in raw_units][:n_layers]

    # Dropout (same rate for all hidden layers to keep it simple)
    dropout = float(np.clip(vector[5], 0.0, 0.5))

    # Activation function
    act_idx = int(np.clip(np.round(vector[6]), 0, len(activation_functions) - 1))
    activation = activation_functions[act_idx]

    # BatchNorm flag
    batch_norm = bool(int(np.clip(np.round(vector[7]), 0, 1)))

    return {
        "n_layers": n_layers,
        "units_per_layer": units,
        "activation": activation,
        "batch_norm": batch_norm,
        "dropout_rate": dropout,
    }


def get_activation_layer(name: str):
    """Return a small set of easy-to-understand activation layers."""
    if name == "relu":
        return tf.keras.layers.Activation("relu")
    if name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(alpha=0.01)
    raise ValueError(f"Unknown activation {name}")
