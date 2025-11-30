import torch, torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import globals.hyperparmeter_optimizer as hyp_optimizer
import globals.model_evaluations as evaluations
import torch.utils.data

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
    activation_name = hyperparameters.get("activation", "relu")
    use_batch_norm = bool(hyperparameters.get("batch_norm", False))
    dropout_rate = float(hyperparameters.get("dropout_rate", 0.0) or 0.0)

    layers = []
    in_dim = int(input_shape)

    for i, out_dim in enumerate(units):
        # Linear
        layers.append(nn.Linear(in_dim, int(out_dim)))
        # BatchNorm (1D for dense features)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(out_dim)))
        # Activation (fresh instance)
        layers.append(get_torch_activation(activation_name))
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
    Retrain the best architecture on all training data (with val_split=0.1),
    then evaluate on the untouched test set at threshold=0.5.
    """
    device = select_device()
    
    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    # Split training into train/val for early stopping (simulating validation_split=0.1)
    # We'll just use a simple split here or use the same logic as Keras validation_split
    # For simplicity, let's use sklearn's train_test_split or manual slicing
    val_size = int(len(X_train) * 0.1)
    train_size = len(X_train) - val_size
    
    # Manual split (assuming shuffled or shuffling here)
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    X_tr, y_tr = X_train_t[train_idx], y_train_t[train_idx]
    X_va, y_va = X_train_t[val_idx], y_train_t[val_idx]
    
    train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_va_dev = X_va.to(device)
            y_va_dev = y_va.to(device)
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
        
    metrics = evaluations.classification_metrics(y_test, y_prob, threshold=0.5)
    return model, metrics
