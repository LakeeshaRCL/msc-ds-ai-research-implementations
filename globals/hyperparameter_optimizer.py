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
        if isinstance(u_val, (float, np.floating)):
             idx = int(np.round(u_val))
        else:
             idx = int(u_val)
        
        # Map index to actual unit count
        idx = np.clip(idx, 0, len(hidden_layer_unit_choices) - 1)
        u_val = hidden_layer_unit_choices[idx]
        
        units_per_layer.append(u_val)
        
        # Activation
        a_val = vector[current_idx + 1]
        
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
