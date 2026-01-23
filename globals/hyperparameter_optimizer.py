import numpy as np
from mealpy.swarm_based import PSO, GWO, FA
from mealpy.utils.space import IntegerVar, FloatVar, BoolVar, CategoricalVar

# supported activation functions (matching torch_gpu_processing)
# supported activation functions (matching torch_gpu_processing)
activation_functions = ["relu", "leaky_relu", "elu", "selu", "gelu", "silu"]
batch_sizes = [256, 512, 1024, 2048]
hidden_layer_unit_choices = list(range(16, 513, 16)) # Up to 512, step 16

def decode_scalar(raw_value, spec):
    val = np.clip(raw_value, spec["low"], spec["high"])
    if spec.get("round"):
        val = int(round(val))
    # handle multiples for integer types
    if "multiple" in spec:
        val = int(round(val / spec["multiple"]) * spec["multiple"])
        val = int(np.clip(val, spec["low"], spec["high"]))
    return val

def get_hyperparameter_bounds_config(min_layers=1, max_layers=4):
    """returns bound config list."""
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
    """decode optimizer vector to mlp params."""
    n_layers = int(np.round(vector[0])) if isinstance(vector[0], (float, np.floating)) else int(vector[0])
    dropout = float(np.clip(vector[1], 0.0, 0.5))
    batch_norm = bool(int(np.round(vector[2]))) if isinstance(vector[2], (float, np.floating)) else bool(vector[2])
    
    units_per_layer = []
    activations = []
    
    current_idx = 3
    for i in range(n_layers):
        if current_idx + 1 >= len(vector): break 
            
        u_val = vector[current_idx]
        if u_val not in hidden_layer_unit_choices:
            if isinstance(u_val, (int, float, np.number)):
                idx = int(np.clip(np.round(u_val), 0, len(hidden_layer_unit_choices) - 1))
                u_val = hidden_layer_unit_choices[idx]
        units_per_layer.append(int(u_val))
        
        a_val = vector[current_idx + 1]
        if a_val not in activation_functions:
            if isinstance(a_val, (int, float, np.number)):
                idx = int(np.clip(np.round(a_val), 0, len(activation_functions) - 1))
                a_val = activation_functions[idx]
        activations.append(a_val)
        
        current_idx += 2
        
    return {
        "n_layers": n_layers,
        "units_per_layer": units_per_layer,
        "activations": activations,
        # legacy compatibility
        "legacy_activation": activations[0] if activations else "relu",
        "legacy_batch_norm": batch_norm,
        "legacy_dropout_rate": dropout,
    }

def run_hyperparam_optimization(algorithm, objective, iterations, population, min_layers=1, max_layers=4):
    """run mealpy optimizer."""
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

    if algorithm.upper() == "PSO":
        optimizer = PSO.OriginalPSO(epoch=iterations, pop_size=population)
    elif algorithm.upper() == "GWO":
        optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=population)
    elif algorithm.upper() == "FA":
        optimizer = FA.OriginalFA(epoch=iterations, pop_size=population)
    else:
        raise ValueError("algorithm must be one of {'PSO','GWO','FA'}")

    best_agent = optimizer.solve(problem)
    best_vec = best_agent.solution
    best_obj = best_agent.target.fitness
    best_hp = optimizer_vectors_to_mlp_hyperparams(best_vec)

    epoch_logs = []
    history_vectors = getattr(optimizer.history, "list_global_best", [])
    history_scores = getattr(optimizer.history, "list_global_best_fitness", [])

    print("\n=== Per-Epoch Global Best Log ===")

    for i in range(len(history_scores)):
        vec_epoch = history_vectors[i]
        f1_epoch = 1.0 - history_scores[i]
        hp_epoch = optimizer_vectors_to_mlp_hyperparams(vec_epoch)
        
        epoch_logs.append({
            "epoch": i + 1,
            "f1": float(f1_epoch),
            "objective": float(history_scores[i]),
            "vector": list(vec_epoch),
            "n_layers": hp_epoch["n_layers"],
            "units": hp_epoch["units_per_layer"],
            "activations": hp_epoch["activations"],
            "batch_norm": hp_epoch["batch_norm"],
            "dropout": hp_epoch["dropout_rate"],
        })

        print(f"Epoch {i+1:02d} | F1={f1_epoch:.4f} | Layers={hp_epoch['n_layers']} | Units={hp_epoch['units_per_layer']} | Dropout={hp_epoch['dropout_rate']:.3f}")

    print("\n=== Final Best Solution ===")
    print(f"Best objective (1 - F1): {best_obj:.6f}  => F1={1 - best_obj:.6f}")
    print("Best hyperparameters:", best_hp)

    return best_vec, best_obj, best_hp, epoch_logs

def get_ae_hyperparameter_bounds_config(min_layers=1, max_layers=4):
    """ae bounds config."""
    # global params
    bounds_config = [
        {"type": "int", "name": "n_encoder_layers", "lb": min_layers, "ub": max_layers},
        {"type": "int", "name": "n_decoder_layers", "lb": min_layers, "ub": max_layers},
        {"type": "categorical", "name": "latent_size", "choices": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]}, 
        {"type": "float", "name": "dropout_rate", "lb": 0.0, "ub": 0.5},
        {"type": "bool", "name": "batch_norm"},
    ]
    
    # layer params
    for i in range(max_layers):
        bounds_config.append({"type": "categorical", "name": f"units_enc_{i+1}", "choices": hidden_layer_unit_choices})
        bounds_config.append({"type": "categorical", "name": f"act_enc_{i+1}", "choices": activation_functions})
        bounds_config.append({"type": "categorical", "name": f"units_dec_{i+1}", "choices": hidden_layer_unit_choices})
        bounds_config.append({"type": "categorical", "name": f"act_dec_{i+1}", "choices": activation_functions})
        
    return bounds_config

def optimizer_vectors_to_ae_hyperparams(vector):
    """decode ae optimizer vector."""
    n_encoder = int(round(decode_scalar(vector[0], {"low": 1, "high": 4})))
    n_decoder = int(round(decode_scalar(vector[1], {"low": 1, "high": 4})))
    
    latent_choices = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    latent_val = vector[2]
    
    if latent_val in latent_choices:
        latent_size = int(latent_val)
    else:
        idx = int(np.clip(round(latent_val), 0, len(latent_choices)-1))
        if latent_val < 0.9 and latent_val > -0.1:
             latent_size = latent_choices[idx]
        else:
             latent_size = min(latent_choices, key=lambda x:abs(x-latent_val))

    dropout = float(np.clip(vector[3], 0.0, 0.5))
    batch_norm = bool(round(vector[4]))
    
    current_idx = 5
    enc_units, enc_acts, dec_units, dec_acts = [], [], [], []
    max_layers_assumed = 8
    
    def get_cat(val, choices):
        if val in choices: return val
        idx = int(np.clip(round(val), 0, len(choices)-1))
        is_numeric = all(isinstance(c, (int, float, np.number)) for c in choices)
        
        if is_numeric:
            if val < len(choices) and val >= -0.5: return choices[idx]
            return min(choices, key=lambda x: abs(x - val))
        else:
            return choices[idx]

    for i in range(max_layers_assumed):
        if i < n_encoder:
            enc_units.append(int(get_cat(vector[current_idx], hidden_layer_unit_choices)))
            enc_acts.append(get_cat(vector[current_idx+1], activation_functions))
            
        if i < n_decoder:
            dec_units.append(int(get_cat(vector[current_idx+2], hidden_layer_unit_choices)))
            dec_acts.append(get_cat(vector[current_idx+3], activation_functions))
            
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
