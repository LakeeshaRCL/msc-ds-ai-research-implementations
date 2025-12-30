import numpy as np
from mealpy.swarm_based import PSO, GWO, FA
from mealpy.utils.space import IntegerVar, FloatVar, BoolVar, CategoricalVar

# Supported activation functions (matching torch_gpu_processing)
activation_functions = ["relu", "leaky_relu"]
batch_sizes = [256, 512, 1024, 2048]
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

    return val

def get_hyperparameter_bounds_config(min_layers=1, max_layers=4):
    """
    Returns a list of dicts defining the bounds for the optimizer vector.
    Each dict has keys: 'type', 'name', 'lb', 'ub', 'choices'.
    """
    bounds_config = [
        {"type": "int", "name": "n_layers", "lb": min_layers, "ub": max_layers},
        {"type": "float", "name": "dropout_rate", "lb": 0.0, "ub": 0.5},
        {"type": "bool", "name": "batch_norm"}, 
    ]
    
    for i in range(max_layers):
        bounds_config.append({
            "type": "categorical", 
            "name": f"units_layer_{i+1}", 
            "choices": hidden_layer_unit_choices
        })
        bounds_config.append({
            "type": "categorical", 
            "name": f"activation_layer_{i+1}", 
            "choices": activation_functions
        })
        
    return bounds_config

def optimizer_vectors_to_mlp_hyperparams(vector):
    """
    Decode the optimizer's continuous vector into MLP architecture.
    vector: Solution vector from Mealpy.
    """
    # 1. Decode fixed header
    # n_layers is at index 0.
    n_layers = vector[0]
    if isinstance(n_layers, (float, np.floating)):
        n_layers = int(np.round(n_layers))
    else:
        n_layers = int(n_layers)
    
    dropout = float(np.clip(vector[1], 0.0, 0.5))
    
    batch_norm_val = vector[2]
    if isinstance(batch_norm_val, (float, np.floating)):
        batch_norm = bool(int(np.round(batch_norm_val)))
    else:
        batch_norm = bool(batch_norm_val)
    
    # 2. Decode layers
    units_per_layer = []
    activations = []
    
    current_idx = 3
    # Only read up to n_layers
    for i in range(n_layers):
        if current_idx + 1 >= len(vector):
            break 
            
        # Units
        u_val = vector[current_idx]
        
        # If the value is not directly in choices, it might be a float index from some optimizers
        if u_val not in hidden_layer_unit_choices:
            if isinstance(u_val, (int, float, np.number)):
                idx = int(np.round(u_val))
                idx = np.clip(idx, 0, len(hidden_layer_unit_choices) - 1)
                u_val = hidden_layer_unit_choices[idx]
        
        units_per_layer.append(int(u_val))
        
        # Activation
        a_val = vector[current_idx + 1]
        
        if a_val not in activation_functions:
            # If numeric, assume index mapping
            if isinstance(a_val, (int, float, np.number)):
                idx = int(np.round(a_val))
                idx = np.clip(idx, 0, len(activation_functions) - 1)
                a_val = activation_functions[idx]
        
        activations.append(a_val)
        
        current_idx += 2
        
    return {
        "n_layers": n_layers,
        "units_per_layer": units_per_layer,
        "activations": activations,
        # Legacy compatibility:  ensure the code remains compatible with older versions of the model building function that might expect
        # a single global activation setting instead of a list of activations for each layer.
        "legacy_activation": activations[0] if activations else "relu",
        "legacy_batch_norm": batch_norm,
        "legacy_dropout_rate": dropout,
    }

def run_hyperparam_optimization(algorithm, objective, iterations, population, min_layers=1, max_layers=4):
    """
    Run the selected mealpy optimizer.
    """
    # Build bounds
    bounds_cfg = get_hyperparameter_bounds_config(min_layers, max_layers)
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
            
    problem = dict(obj_func=objective, bounds=optimizer_bounds, minmax="min", log_to=None)

    # Select optimizer
    if algorithm.upper() == "PSO":
        optimizer = PSO.OriginalPSO(epoch=iterations, pop_size=population)
    elif algorithm.upper() == "GWO":
        optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=population)
    elif algorithm.upper() == "FA":
        optimizer = FA.OriginalFA(epoch=iterations, pop_size=population)
    else:
        raise ValueError("algorithm must be one of {'PSO','GWO','FA'}")

    # Run optimization
    best_agent = optimizer.solve(problem)
    best_vec = best_agent.solution
    best_obj = best_agent.target.fitness
    best_hp = optimizer_vectors_to_mlp_hyperparams(best_vec)

    # Collect per-epoch history
    epoch_logs = []
    try:
        history_vectors = getattr(optimizer.history, "list_global_best", [])
        history_scores = getattr(optimizer.history, "list_global_best_fitness", [])
    except AttributeError:
        history_vectors, history_scores = [], []

    total_epochs = len(history_scores)

    print("\n=== Per-Epoch Global Best Log ===")

    for i in range(total_epochs):
        vec_epoch = history_vectors[i]
        obj_epoch = history_scores[i]
        f1_epoch = 1.0 - obj_epoch  # Because objective = 1 - F1
        hp_epoch = optimizer_vectors_to_mlp_hyperparams(vec_epoch)
        
        log_record = {
            "epoch": i + 1,
            "f1": float(f1_epoch),
            "objective": float(obj_epoch),
            "vector": [v for v in vec_epoch], # keep raw types
            "n_layers": hp_epoch["n_layers"],
            "units": hp_epoch["units_per_layer"],
            "activations": hp_epoch["activations"],
            "batch_norm": hp_epoch["batch_norm"],
            "dropout": hp_epoch["dropout_rate"],
        }
        epoch_logs.append(log_record)

        print(f"Epoch {i+1:02d}/{total_epochs} | F1={f1_epoch:.4f} | Layers={hp_epoch['n_layers']} "
              f"| Units={hp_epoch['units_per_layer']} | Acts={hp_epoch['activations']} "
              f"| BN={hp_epoch['batch_norm']} | Dropout={hp_epoch['dropout_rate']:.3f}")

    print("\n=== Final Best Solution ===")
    print(f"Best objective (1 - F1): {best_obj:.6f}  => F1={1 - best_obj:.6f}")
    print("Best hyperparameters:", best_hp)

    return best_vec, best_obj, best_hp, epoch_logs

def get_ae_hyperparameter_bounds_config(min_layers=1, max_layers=4):
    """
    Returns a list of dicts defining the bounds for the AE optimizer vector.
    Parameters:
        min_layers, max_layers: Bounds for *encoder* and *decoder* depth separately.
    """
    # 1. Global Arch Params
    bounds_config = [
        {"type": "int", "name": "n_encoder_layers", "lb": min_layers, "ub": max_layers},
        {"type": "int", "name": "n_decoder_layers", "lb": min_layers, "ub": max_layers},
        {"type": "categorical", "name": "latent_size", "choices": [4, 8, 12, 16, 20, 24, 32, 64]}, # Bottleneck
        {"type": "float", "name": "dropout_rate", "lb": 0.0, "ub": 0.5},
        {"type": "bool", "name": "batch_norm"},
    ]
    
    # 2. Dynamic Layer Params (Allocating max slots for both Enc and Dec)
    # We will use the same pool of choices as MLP
    for i in range(max_layers):
        # Encoder Layer i
        bounds_config.append({
            "type": "categorical", 
            "name": f"units_enc_{i+1}", 
            "choices": hidden_layer_unit_choices
        })
        bounds_config.append({
            "type": "categorical", 
            "name": f"act_enc_{i+1}", 
            "choices": activation_functions
        })
        
        # Decoder Layer i
        bounds_config.append({
            "type": "categorical", 
            "name": f"units_dec_{i+1}", 
            "choices": hidden_layer_unit_choices
        })
        bounds_config.append({
            "type": "categorical", 
            "name": f"act_dec_{i+1}", 
            "choices": activation_functions
        })
        
    return bounds_config

def optimizer_vectors_to_ae_hyperparams(vector):
    """
    Decode the optimizer's continuous vector into AE architecture.
    """
    # 1. Decode Fixed Params
    # Indices: 0:n_enc, 1:n_dec, 2:latent, 3:dropout, 4:bn
    
    n_encoder = int(round(decode_scalar(vector[0], {"low": 1, "high": 4}))) # Bounds assumed from default
    n_decoder = int(round(decode_scalar(vector[1], {"low": 1, "high": 4})))
    
    # Latent size (Categorical)
    # If vector value is not in choices, treat as index or map closest
    latent_choices = [4, 8, 12, 16, 20, 24, 32, 64]
    
    latent_val = vector[2]
    if latent_val in latent_choices:
        latent_size = int(latent_val)
    else:
        # Map float to index
        idx = int(np.clip(round(latent_val), 0, len(latent_choices)-1))
        # Depending on optimizer, it might pass the value directly if it learned it, or an index.
        # Mealpy usually passes the value for categorical if configured right, but let's be safe.
        # Ideally we should use the same decoding logic as MLP which seemed to handle it dynamically
        # But here I'll just be robust.
        if latent_val < 0.9 and latent_val > -0.1: # Likely normalized or index? safely assume index if small
             latent_size = latent_choices[idx]
        else:
             # Try to find closest
             latent_size = min(latent_choices, key=lambda x:abs(x-latent_val))

    dropout = float(np.clip(vector[3], 0.0, 0.5))
    batch_norm = bool(round(vector[4]))
    
    # 2. Decode Dynamic Layers
    # Start after fixed params
    current_idx = 5
    
    enc_units = []
    enc_acts = []
    dec_units = []
    dec_acts = []
    
    # Max layers assumed 4 for decoding loop safety, or we just loop until we run out of vector
    # Detailed mapping: For each of MAX layers, we have (Units, Act) for Enc, then (Units, Act) for Dec?
    # No, the bounds config order was: Enc1, Act1, Dec1, Act1 ... (interleaved? No, loop says Enc then Dec)
    # Loop was: for i in range(max): append Enc_i, append Act_i, append Dec_i, append Act_i
    
    # We need to know max_layers to decode correctly if we want to skip unused slots,
    # OR we just read the slots we need based on n_encoder/n_decoder and ignore the rest?
    # BUT the vector position is fixed. We MUST know the stride.
    # We will assume max_layers=4 as per default in bounds config.
    max_layers_assumed = 4 
    
    # Function to safe decode categorical
    def get_cat(val, choices):
        # 1. Exact match check
        if val in choices: return val
        
        # 2. Index Fallback
        # Map float to valid index range
        idx = int(np.clip(round(val), 0, len(choices)-1))
        
        # 3. Check if choices are numeric
        is_numeric = all(isinstance(c, (int, float, np.number)) for c in choices)
        
        if is_numeric:
            # If numeric, we can use the "closest value" logic
            # Be careful if val is clearly an index vs a value
            # Heuristic: If val is float (e.g. 0.5) and choices are large ints [16, 32], assume index.
            # But simpler is to assume Mealpy returns the mapped value if it can, or the raw bounds if not.
            # For CategoricalVar, Mealpy often returns the VALUE if we access the variable, but the SOLUTION VECTOR is usually numbers.
            # If the solution vector corresponds to the bound index, then 'val' is an index.
            # Let's try the index first.
            if val < len(choices) and val >= -0.5 and (val % 1 != 0 or val < 1.0):
                 return choices[idx]
                 
            # Otherwise try distance
            return min(choices, key=lambda x: abs(x - val))
        else:
            # If NOT numeric (e.g. strings), we MUST rely on index
            return choices[idx]

    for i in range(max_layers_assumed):
        # Indices in loop: 
        # Enc_Units: current_idx
        # Enc_Act: current_idx + 1
        # Dec_Units: current_idx + 2
        # Dec_Act: current_idx + 3
        
        # Encoder Layer i
        if i < n_encoder:
            u = get_cat(vector[current_idx], hidden_layer_unit_choices)
            a = get_cat(vector[current_idx+1], activation_functions)
            enc_units.append(int(u))
            enc_acts.append(a)
            
        # Decoder Layer i
        if i < n_decoder:
            u = get_cat(vector[current_idx+2], hidden_layer_unit_choices)
            a = get_cat(vector[current_idx+3], activation_functions)
            dec_units.append(int(u))
            dec_acts.append(a)
            
        current_idx += 4
        
    return {
        "n_encoder_layers": n_encoder,
        "n_decoder_layers": n_decoder,
        "latent_size": latent_size,
        "encoder_units": enc_units,
        "encoder_activations": enc_acts,
        "decoder_units": dec_units,
        "decoder_activations": dec_acts,
        "dropout_rate": dropout,
        "batch_norm": batch_norm
    }
