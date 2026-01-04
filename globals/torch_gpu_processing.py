import torch, torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import globals.hyperparameter_optimizer as hyp_optimizer
import globals.model_evaluations as evaluations
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import gc


# Suppress DirectML CPU fallback warnings explicitly for cleaner logs
# Using regex to ensure we match the message correctly
warnings.filterwarnings("ignore", message=r".*aten::lerp.*")
warnings.filterwarnings("ignore", message=r".*aten::elu.*")

def select_device():
    return torch_directml.device()

def test_gpu_processing():
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

def batched_inference(model: nn.Module, X_tensor: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """
    Perform inference in batches to avoid OOM on large datasets.
    Returns: numpy array of probabilities.
    """
    model.eval()
    probs_list = []
    n_samples = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i : i + batch_size]
            batch = X_tensor[i : i + batch_size]
            outputs = model(batch)
            probs_list.append(outputs)
            
    if not probs_list:
        return np.array([])
        
    return torch.cat(probs_list).cpu().numpy().ravel()

def batched_validation_loss(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor, 
                          loss_fn: nn.Module, batch_size: int = 1024) -> float:
    """
    Compute validation loss in batches.
    Returns: average loss (scalar).
    """
    model.eval()
    # Initialize accumulator on the correct device
    total_loss = torch.tensor(0.0, device=X_tensor.device)
    n_samples = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = X_tensor[i : i + batch_size]
            y_batch = y_tensor[i : i + batch_size]
            
            outputs = model(X_batch)
            # Loss function returns a tensor scalar (mean). 
            # Accumulate on device.
            batch_loss = loss_fn(outputs, y_batch)
            total_loss += batch_loss * len(X_batch)
            
    # Single sync at the end
    return total_loss.item() / n_samples


def test_direct_ml_processing():
    print("torch:", torch.__version__)
    d = torch_directml.device()
    print("DirectML device:", d)
    x = torch.randn(4, device=d)
    print("Tensor device:", x.device)


def get_torch_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    # fallback
    return nn.ReLU()


def build_mlp_model(hyperparameters, input_shape, lr=0.001):
    """
    Build a PyTorch MLP (functional style) and return model, optimizer, loss_fn, device.

    Args:
      hyperparameters (dict): keys "units_per_layer", "activation", "batch_norm", "dropout_rate"
      input_shape (int): number of input features
      lr (float): learning rate


    Returns:
      model (nn.Module): the MLP model (moved to device). Outputs raw logits (no sigmoid).
      optimizer (torch.optim.Optimizer): Adam optimizer for model.parameters()
      loss_fn (callable): nn.BCEWithLogitsLoss()

    """
    # Device selection
    device = select_device()


    units = list(hyperparameters.get("units_per_layer", []))
    # Support per-layer activations
    activations = list(hyperparameters.get("activations", []))
    if not activations:
        # Fallback to global activation if per-layer list not provided
        global_act = hyperparameters.get("legacy_activation", "relu")
        activations = [global_act] * len(units)
        
    use_batch_norm = bool(hyperparameters.get("legacy_batch_norm", False))
    dropout_rate = float(hyperparameters.get("legacy_dropout_rate", 0.0) or 0.0)

    layers = []
    in_dim = int(input_shape)

    for i, out_dim in enumerate(units):
        # Linear
        layers.append(nn.Linear(in_dim, int(out_dim)))
        # BatchNorm (1D for dense features)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(out_dim)))
        # Activation (fresh instance)
        # activations list was padded/prepared above
        act_name = activations[i] if i < len(activations) else "relu"
        layers.append(get_torch_activation(act_name))
        # Dropout
        if dropout_rate and dropout_rate > 0.0:
            layers.append(nn.Dropout(p=float(dropout_rate)))
        in_dim = int(out_dim)

    # Final linear -> raw logits
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    # Move model to device (DirectML expects float32)
    model.to(device, dtype=torch.float32)

    # display model summary (simple print to avoid DirectML/summary conflicts)
    print(model)

    return model


class ManualBCELoss(nn.Module):
    def __init__(self, pos_weight=None, epsilon=1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip to prevent log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        if self.pos_weight is not None:
            # Weighted BCE: -[pos_weight * y * log(p) + (1-y) * log(1-p)]
            loss = -(self.pos_weight * y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        else:
            # Standard BCE
            loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        return torch.mean(loss)


def set_optimizer_objective(X_train, y_train, X_val, y_val, max_epochs, batch_size, seed, early_stopping_patience):
    """
    Build the objective: minimize (1 - F1 on validation set).
    """
    # Ensure inputs are numpy arrays
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(X_val, np.ndarray):
        X_val = np.array(X_val)
    if not isinstance(y_val, np.ndarray):
        y_val = np.array(y_val)

    device = select_device()
    print(f"Optimizer using device: {device}")
    
    # Ensure labels are 2D (N, 1) correctly regardless of input type
    y_train_flat = np.array(y_train).flatten()
    y_val_flat = np.array(y_val).flatten()

    # PERFORMANCE OPTIMIZATION: Downsample Validation Set for Hyperparameter Tuning
    # Evaluating 118k samples every epoch is expensive. Limit to 30k for optimization speed.
    if len(X_val) > 10000:
        print(f"Downsampling validation set from {len(X_val)} to 10000 for optimization speed.")
        np.random.seed(seed)
        val_indices = np.random.choice(len(X_val), size=10000, replace=False)
        X_val = X_val[val_indices]
        y_val_flat = y_val_flat[val_indices]
        
    print(f"Final Optimization Validation Size: {len(X_val)}")
    
    # PERFORMANCE FIX: Move ENTIRE dataset to GPU upfront
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    print(f"DEBUG: Optimizer Data Shapes -> Train: {X_train_t.shape}, Val: {X_val_t.shape}")

    
    # Pre-calculate class weights robustly
    num_pos = (y_train_flat == 1).sum()
    num_neg = (y_train_flat == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    
    # For manual batching
    n_samples = len(X_train_t)

    def obj(vec):
        hp = hyp_optimizer.optimizer_vectors_to_mlp_hyperparams(vec)
        
        # Build model
        model = build_mlp_model(hp, X_train.shape[1], lr=0.001)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
        loss_fn = ManualBCELoss(pos_weight=pos_weight)
        
        # Training loop w/ early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        best_model_state = None
        
        start_time = time.time()
        for epoch in range(max_epochs):

            model.train()
            
            # Manual shuffling and batching to avoid DataLoader overhead/GPU item access
            # Generate random permutation on CPU, then use to index GPU tensors
            perm = torch.randperm(n_samples) # CPU tensor is fine for indexing
            
            for i in range(0, n_samples, batch_size):
                indices = perm[i : i + batch_size]
                
                # Check for last batch
                if len(indices) == 0:
                    continue
                    
                # Indexing GPU tensor with CPU indices works in PyTorch and is efficient (gather)
                X_batch = X_train_t[indices]
                y_batch = y_train_t[indices]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            # Validation (Batched)
            val_loss = batched_validation_loss(model, X_val_t, y_val_t, loss_fn, batch_size)
            
            
            # Print dot for epoch progress (visual feedback)
            print(".", end="", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        train_duration = time.time() - start_time
        print(f" {train_duration:.1f}s]", end="")

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Evaluate F1 on validation set (Batched)
        y_prob = batched_inference(model, X_val_t, batch_size)
            
        # Use optimal threshold for the objective
        # This allows the optimizer to find models that have high potential F1, 
        # even if they are not perfectly calibrated to 0.5 yet.
        _, best_f1, _ = evaluations.find_optimal_threshold(y_val_t.cpu().numpy(), y_prob)
        
        return 1.0 - float(best_f1)

    # Force garbage collection to help DirectML
    import gc
    gc.collect()



    return obj


def retrain_and_evaluate(best_hp, X_train, y_train, X_test, y_test, batch_size,
                         max_epochs=40, early_stopping_patience=8):
    """
    Retrain the best architecture on all training data (X_train),
    and use X_test for validation/early stopping.
    Finally evaluate on X_test.
    """
    
    # Handle tuple input from optimization
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        # Optimization result is typically (vector, objective, hp_dict, logs)
        print("Extracting hyperparameters from optimization result tuple...")
        best_hp = best_hp[2]

    device = select_device()
    print(f"Retraining on device: {device}")
    
    # Convert data to tensors and move to GPU upfront
    y_train_flat = np.array(y_train).flatten()
    y_test_flat = np.array(y_test).flatten()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Train on Full X_train
    # Use X_test as validation set for early stopping
    
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
    # Calculate class weights from training data
    num_pos = (y_train_flat == 1).sum()
    num_neg = (y_train_flat == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    loss_fn = ManualBCELoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    n_samples = len(X_train_t)
    
    for epoch in range(max_epochs):
        model.train()
        
        # Manual batching
        perm = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            indices = perm[i : i + batch_size]
            if len(indices) == 0: continue
            
            X_batch = X_train_t[indices]
            y_batch = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation on TEST SET (Batched)
        val_loss = batched_validation_loss(model, X_test_t, y_test_t, loss_fn, batch_size)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Evaluate on test set (Batched)
    y_prob = batched_inference(model, X_test_t, batch_size)
        
    
    # Calculate metrics with default threshold 0.5
    metrics = evaluations.classification_metrics(y_test, y_prob, threshold=0.5)
    
    # Find optimal threshold
    best_thresh, best_f1, best_metrics = evaluations.find_optimal_threshold(y_test, y_prob)
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Threshold: {best_thresh:.4f}")
    print(f"  F1 Score:  {best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    
    # Add optimal metrics to the return dictionary
    metrics.update({
        "optimal_threshold": best_thresh,
        "optimal_f1": best_f1,
        "optimal_precision": best_metrics['precision'],
        "optimal_recall": best_metrics['recall']
    })
    
    return model, metrics


def train_final_model(best_hp, X_train, y_train, X_test, y_test, batch_size, 
                      max_epochs=50, early_stopping_patience=10, save_path=None):
    """
    Train the final model using the best hyperparameters found during optimization.
    This function is separate from optimization helpers to allow for specific
    final-model configurations (e.g., saving, extensive logging).
    
    Args:
        best_hp (dict): Best hyperparameters.
        X_train, y_train: Training data.
        X_test, y_test: Validation/Test data used for early stopping and evaluation.
        batch_size (int): Batch size.
        max_epochs (int): Maximum training epochs.
        early_stopping_patience (int): Patience for early stopping.
        save_path (str, optional): Path to save the best model state dict.
        
    Returns:
        model: Trained PyTorch model.
        metrics: Dictionary of evaluation metrics.
    """
    print("="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    # Handle tuple input from optimization if passed directly
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        print("Extracting hyperparameters from optimization result tuple...")
        best_hp = best_hp[2]
        
    print(f"Hyperparameters: {best_hp}")
    
    device = select_device()
    print(f"Training on device: {device}")
    
    # Convert data to tensors and move to GPU upfront
    y_train_flat = np.array(y_train).flatten()
    y_test_flat = np.array(y_test).flatten()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Build model
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
    # Class weights
    num_pos = (y_train_flat == 1).sum()
    num_neg = (y_train_flat == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    print(f"Using positive class weight: {pos_weight:.4f}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    loss_fn = ManualBCELoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training for {max_epochs} epochs (patience={early_stopping_patience})...")
    
    n_samples = len(X_train_t)
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_accum = 0.0
        batches = 0
        
        # Manual batching
        perm = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            indices = perm[i : i + batch_size]
            if len(indices) == 0: continue
            
            X_batch = X_train_t[indices]
            y_batch = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            batches += 1
            
        avg_train_loss = train_loss_accum / batches if batches > 0 else 0
        
        # Validation (Batched)
        val_loss = batched_validation_loss(model, X_test_t, y_test_t, loss_fn, batch_size)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f} (New Best)")
        else:
            patience_counter += 1
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Restored best model weight state.")
        
    # Save model if path provided
    if save_path:
        try:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")

    # Final Evaluation (Batched)
    y_prob = batched_inference(model, X_test_t, batch_size)
        
    print("\nEvaluating final model...")
    
    # 1. Standard Metrics (0.5 threshold)
    metrics = evaluations.classification_metrics(y_test, y_prob, threshold=0.5)
    
    # 2. Optimal Threshold Metrics
    best_thresh, best_f1, best_metrics = evaluations.find_optimal_threshold(y_test, y_prob)
    
    metrics.update({
        "optimal_threshold": best_thresh,
        "optimal_f1": best_f1,
        "optimal_precision": best_metrics['precision'],
        "optimal_recall": best_metrics['recall']
    })
    
    print(f"\nResults @ Optimal Threshold ({best_thresh:.4f}):")
    print(f"  F1 Score:  {best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    
    return model, metrics


# ==================================================================================
# AUTOENCODER IMPLEMENTATION
# ==================================================================================

def build_ae_model(hyperparameters, input_shape):
    """
    Builds an Autoencoder model (Encoder -> Bottleneck -> Decoder).
    Output dimension == Input dimension (input_shape).
    """
    device = select_device()
    
    # Extract params
    n_enc = int(hyperparameters.get("n_encoder_layers", 1))
    n_dec = int(hyperparameters.get("n_decoder_layers", 1))
    latent_dim = int(hyperparameters.get("latent_size", 8))
    
    enc_units = hyperparameters.get("encoder_units", [])
    enc_acts = hyperparameters.get("encoder_activations", [])
    dec_units = hyperparameters.get("decoder_units", [])
    dec_acts = hyperparameters.get("decoder_activations", [])
    
    dropout = float(hyperparameters.get("dropout_rate", 0.0))
    use_bn = bool(hyperparameters.get("batch_norm", False))
    
    layers = []
    
    # --- ENCODER ---
    in_dim = int(input_shape)
    for i in range(n_enc):
        out_dim = enc_units[i] if i < len(enc_units) else 64
        act_name = enc_acts[i] if i < len(enc_acts) else "relu"
        
        layers.append(nn.Linear(in_dim, out_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(get_torch_activation(act_name))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        in_dim = out_dim
        
    # --- BOTTLENECK ---
    layers.append(nn.Linear(in_dim, latent_dim))
    layers.append(nn.ReLU()) 
    
    # --- DECODER ---
    in_dim = latent_dim
    for i in range(n_dec):
        out_dim = dec_units[i] if i < len(dec_units) else 64
        act_name = dec_acts[i] if i < len(dec_acts) else "relu"
        
        layers.append(nn.Linear(in_dim, out_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(get_torch_activation(act_name))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        in_dim = out_dim
        
    # --- OUTPUT LAYER ---
    layers.append(nn.Linear(in_dim, int(input_shape)))
    # Output activation: Identity (Linear) + MSE Loss.
    
    model = nn.Sequential(*layers)
    model.to(device, dtype=torch.float32)
    # print(model) # Reduced verbosity for optimization
    return model

def batched_reconstruction_loss(model, X_tensor, batch_size=1024):
    """Calculate MSE reconstruction loss in batches."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = torch.tensor(0.0, device=X_tensor.device)
    n = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_tensor[i:i+batch_size]
            recon = model(batch)
            loss = criterion(recon, batch)
            total_loss += loss * len(batch)
            
    return total_loss.item() / n

def get_reconstruction_errors(model, X_tensor, batch_size=1024):
    """Return squared error per sample: mean((X - Rec)^2, axis=1)"""
    model.eval()
    errors_list = []
    n = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_tensor[i:i+batch_size]
            recon = model(batch)
            # squared difference per feature
            # mean over features (axis 1)
            batch_errors = torch.mean((batch - recon)**2, dim=1)
            errors_list.append(batch_errors)
            
    return torch.cat(errors_list).cpu().numpy()

def set_ae_optimizer_objective(X_train, y_train, X_val, y_val, max_epochs, batch_size, seed, early_stopping_patience, force_cpu=True):
    """
    AE Objective: 
    1. Train on X_train (Reconstruction).
    2. Evaluate on X_val (Reconstruction Error thresholding -> F1).
    3. Return (1 - Optimal F1).
    """
    if force_cpu:
        device = torch.device('cpu')
        print("Optimizer using CPU for stability on small samples.")
    else:
        device = select_device()
        
    display_device = device
    
    # 1. Data Prep
    # Validation downsampling for speed
    if len(X_val) > 10000:
        np.random.seed(seed)
        idx = np.random.choice(len(X_val), size=10000, replace=False)
        X_val_opt = X_val[idx]
        y_val_opt = np.array(y_val).flatten()[idx]
    else:
        X_val_opt = X_val
        y_val_opt = np.array(y_val).flatten()
        
    # Prepare Training DataLoader (CPU -> GPU stream)
    # Using pin_memory=True enables faster host-to-device copy
    # Construct TensorDataset from numpy
    X_train_tensor = torch.from_numpy(X_train).float()
    train_dataset = TensorDataset(X_train_tensor)
    
    # Validation data can be smaller, so we might keep it on GPU or use loader
    # For consistency and memory safety, let's use a loader or just move chunks
    # Given optimization loop checks validation every epoch, let's pre-move validation to GPU if it fits
    # 10k samples * 26 features * 4 bytes ~= 1MB. This is safe to keep on GPU.
    X_val_t = torch.tensor(X_val_opt, dtype=torch.float32).to(device)
    # y_val_opt is needed for F1 (CPU side usually for sklearn, but find_optimal_threshold might need numpy)
    
    def obj(vec):
        # Explicit garbage collection to prevent memory buildup in trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        hp = hyp_optimizer.optimizer_vectors_to_ae_hyperparams(vec)
        
        # Build
        model = build_ae_model(hp, X_train.shape[1])
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create DataLoader per trial (shuffle changes)
        # num_workers=0 for Windows DirectML safety (multiprocessing can be tricky)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=False # Disabled for DirectML stability
        )
        
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        # Train Loop
        for epoch in range(max_epochs):
            epoch_start = time.time()
            model.train()
            
            for batch_data in train_loader:
                # DataLoader returns a list/tuple of [features, target] or just [features]
                # TensorDataset(X) returns (X,)
                batch_x = batch_data[0].to(device, non_blocking=False)
                
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                
            # Validation (MSE)
            # X_val_t is already on device
            val_loss = batched_reconstruction_loss(model, X_val_t, batch_size)
            
            # Time tracking
            epoch_duration = time.time() - epoch_start
            print(f"[{epoch+1}:{epoch_duration:.1f}s]", end="", flush=True)
            # print(".", end="", flush=True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
        
        print("]")
        
        # Restore best reconstruction model
        if best_state:
            model.load_state_dict(best_state)
            
        # Evaluation: Find Best F1 by Thresholding Reconstruction Error
        # Ensure we don't leak GPU memory here
        with torch.no_grad():
            recon_errors = get_reconstruction_errors(model, X_val_t, batch_size)
        
        _, best_f1, _ = evaluations.find_optimal_threshold(y_val_opt, recon_errors)
        
        # Clean up model
        del model
        del optimizer
        
        return 1.0 - best_f1
        
    return obj

def train_final_ae_model(best_hp, X_train, y_train, X_test, y_test, batch_size, max_epochs=50, save_path=None):
    """
    Train final AE on full data, evaluate on Test.
    """
    print("="*60)
    print("FINAL AE TRAINING")
    print("="*60)
    
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        best_hp = best_hp[2]
    
    # device = select_device() # select_device called in build_ae_model
    display_device = torch_directml.device() # or use select_device just to print
    
    # Prep DataLoader
    X_train_tensor = torch.from_numpy(X_train).float()
    train_dataset = TensorDataset(X_train_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(display_device)
    
    model = build_ae_model(best_hp, X_train.shape[1])
    # model already moved to device in build_ae_model
    device = next(model.parameters()).device
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    patience_limit = 10
    
    print(f"Training AE for max {max_epochs} epochs on {device}...")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        
        for batch_data in train_loader:
            batch_x = batch_data[0].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
        avg_train_loss = train_loss / batches if batches > 0 else 0
        
        # Validation (Reconstruction Loss on Test/Validation set)
        val_loss = batched_reconstruction_loss(model, X_test_t, batch_size)
        
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping triggered.")
                break
                
    if best_state:
        model.load_state_dict(best_state)
        
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
        

        
    # Evaluation
    print("\nEvaluating AE Classification Performance...")
    recon_errors = get_reconstruction_errors(model, X_test_t, batch_size)
    y_test_flat = np.array(y_test).flatten()
    
    thresh, f1, metrics = evaluations.find_optimal_threshold(y_test_flat, recon_errors)
    
    print(f"Optimal Reconstruction Threshold: {thresh:.6f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test_flat, recon_errors)
        print(f"ROC AUC: {auc:.4f}")
        metrics['roc_auc'] = auc
    except: pass
    
    metrics['optimal_threshold'] = thresh
    return model, metrics
