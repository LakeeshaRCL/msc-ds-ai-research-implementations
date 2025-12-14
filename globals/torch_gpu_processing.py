import torch, torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import globals.hyperparameter_optimizer as hyp_optimizer
import globals.model_evaluations as evaluations
import torch.utils.data
import warnings

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


def set_optimizer_objective(X, y, cv, max_epochs, batch_size, seed, early_stopping_patience):
    """
    Build the objective: minimize (1 - mean F1 across CV folds).
    """
    # Ensure X and y are numpy arrays
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    splits = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed).split(np.zeros_like(y), y))
    device = select_device()

    def obj(vec):
        hp = hyp_optimizer.optimizer_vectors_to_mlp_hyperparams(vec)
        f1_scores = []
        
        for tr_idx, va_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            # Convert to tensors
            X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
            X_va_t = torch.tensor(X_va, dtype=torch.float32)
            y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

            # Create DataLoaders
            train_dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Build model
            model = build_mlp_model(hp, X.shape[1], lr=0.001)
            model.to(device)
            
            # Calculate class weights for this fold
            num_pos = (y_tr == 1).sum()
            num_neg = (y_tr == 0).sum()
            pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
            
            optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
            loss_fn = ManualBCELoss(pos_weight=pos_weight)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(max_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    X_va_dev = X_va_t.to(device)
                    y_va_dev = y_va_t.to(device)
                    val_outputs = model(X_va_dev)
                    val_loss = loss_fn(val_outputs, y_va_dev).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
            
            # Restore best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Evaluate F1
            model.eval()
            with torch.no_grad():
                X_va_dev = X_va_t.to(device)
                y_prob = model(X_va_dev).cpu().numpy().ravel()
                
            f1_scores.append(evaluations.classification_metrics(y_va, y_prob, threshold=0.5)["f1"])

        return 1.0 - float(np.mean(f1_scores))

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
    
    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Train on Full X_train
    # Use X_test as validation set for early stopping
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
    # Calculate class weights from training data
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    loss_fn = ManualBCELoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation on TEST SET
        model.eval()
        with torch.no_grad():
            X_va_dev = X_test_t.to(device)
            y_va_dev = y_test_t.to(device)
            val_outputs = model(X_va_dev)
            val_loss = loss_fn(val_outputs, y_va_dev).item()
            
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
        
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_t.to(device)
        y_prob = model(X_test_dev).cpu().numpy().ravel()
        
    
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
    
    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Dataset and Loader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Build model
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
    # Class weights
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Using positive class weight: {pos_weight:.4f}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    loss_fn = ManualBCELoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training for {max_epochs} epochs (patience={early_stopping_patience})...")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_accum = 0.0
        batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            batches += 1
            
        avg_train_loss = train_loss_accum / batches if batches > 0 else 0
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_va_dev = X_test_t.to(device)
            y_va_dev = y_test_t.to(device)
            val_outputs = model(X_va_dev)
            val_loss = loss_fn(val_outputs, y_va_dev).item()
            
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

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_t.to(device)
        y_prob = model(X_test_dev).cpu().numpy().ravel()
        
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
