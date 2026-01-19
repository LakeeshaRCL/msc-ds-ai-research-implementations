import torch
import torch_directml
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


# suppress directml cpu fallback warnings
warnings.filterwarnings("ignore", message=r".*aten::lerp.*")
warnings.filterwarnings("ignore", message=r".*aten::elu.*")

def select_device():
    """selects best available device (cuda > directml > cpu)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device('cpu')

def test_direct_ml_processing():
    """checks if directml is working correctly."""
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"DirectML device: {device}")
        
        # simple tensor op
        x = torch.tensor([1.0, 2.0]).to(device)
        y = x * 2
        print(f"Test operation successful: {y.cpu().numpy()}")
        return True
    except Exception as e:
        print(f"DirectML check failed: {e}")
        return False

def test_gpu_processing():
    print(torch.__version__)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

def batched_inference(model: nn.Module, X_tensor: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """perform inference in batches."""
    model.eval()
    probs_list = []
    n_samples = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i : i + batch_size]
            outputs = model(batch)
            probs_list.append(outputs)
            
    if not probs_list:
        return np.array([])
        
    return torch.cat(probs_list).cpu().numpy().ravel()

def batched_validation_loss(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor, loss_fn: nn.Module, batch_size: int = 1024) -> float:
    """compute validation loss in batches."""
    model.eval()
    total_loss = torch.tensor(0.0, device=X_tensor.device)
    n_samples = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = X_tensor[i : i + batch_size]
            y_batch = y_tensor[i : i + batch_size]
            
            outputs = model(X_batch)
            # loss function returns mean, multiply by batch size to accumulate
            batch_loss = loss_fn(outputs, y_batch)
            total_loss += batch_loss * len(X_batch)
            
    return total_loss.item() / n_samples

def get_torch_activation(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "elu": return nn.ELU()
    if name == "selu": return nn.SELU()
    if name == "leaky_relu": return nn.LeakyReLU(negative_slope=0.01)
    return nn.ReLU()


def build_mlp_model(hyperparameters, input_shape, lr=0.001):
    """build pytorch mlp model."""
    device = select_device()

    units = list(hyperparameters.get("units_per_layer", []))
    activations = list(hyperparameters.get("activations", []))
    
    # fallback if per-layer missing
    if not activations:
        global_act = hyperparameters.get("legacy_activation", "relu")
        activations = [global_act] * len(units)
        
    use_batch_norm = bool(hyperparameters.get("legacy_batch_norm", False))
    dropout_rate = float(hyperparameters.get("legacy_dropout_rate", 0.0) or 0.0)

    layers = []
    in_dim = int(input_shape)

    for i, out_dim in enumerate(units):
        layers.append(nn.Linear(in_dim, int(out_dim)))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(out_dim)))
            
        act_name = activations[i] if i < len(activations) else "relu"
        layers.append(get_torch_activation(act_name))
        
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=float(dropout_rate)))
        in_dim = int(out_dim)

    # final layer (logits)
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    model.to(device, dtype=torch.float32)
    # print(model) 

    return model


class ManualBCELoss(nn.Module):
    def __init__(self, pos_weight=None, epsilon=1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # clip to prevent log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        if self.pos_weight is not None:
            # weighted bce
            loss = -(self.pos_weight * y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        else:
            loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        return torch.mean(loss)


def set_optimizer_objective(X_train, y_train, X_val, y_val, max_epochs, batch_size, seed, early_stopping_patience):
    """build objective function: minimize (1 - f1)."""
    # ensure inputs are numpy arrays
    X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
    y_val = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val

    device = select_device()
    print(f"Optimizer using device: {device}")
    
    y_train_flat = y_train.flatten()
    y_val_flat = y_val.flatten()

    # downsample validation set for speed during optimization if too large
    if len(X_val) > 10000:
        print(f"Downsampling validation set from {len(X_val)} to 10000.")
        from sklearn.model_selection import train_test_split
        # stratify to keep class distribution
        X_val, _, y_val_flat, _ = train_test_split(
            X_val, y_val_flat, train_size=10000, stratify=y_val_flat, random_state=seed
        )
    
    print(f"Final Optimization Validation Size: {len(X_val)}")
    
    # move data to gpu upfront
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # pre-calculate class weights
    num_pos = (y_train_flat == 1).sum()
    num_neg = (y_train_flat == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    
    n_samples = len(X_train_t)

    def obj(vec):
        hp = hyp_optimizer.optimizer_vectors_to_mlp_hyperparams(vec)
        
        model = build_mlp_model(hp, X_train.shape[1], lr=0.001)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
        loss_fn = ManualBCELoss(pos_weight=pos_weight)
        
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience_counter_loss = 0
        patience_counter_f1 = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            try:
                model.train()
                
                # manual shuffle
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
                
                # validation
                val_loss = batched_validation_loss(model, X_val_t, y_val_t, loss_fn, batch_size)
                print(".", end="", flush=True)

                # early stopping logic
                loss_improved = val_loss < best_val_loss
                
                if loss_improved:
                    best_val_loss = val_loss
                    patience_counter_loss = 0
                    
                    # check f1 if loss improved
                    y_prob_epoch = batched_inference(model, X_val_t, batch_size)
                    _, val_f1, _ = evaluations.find_optimal_threshold(y_val_t.cpu().numpy(), y_prob_epoch)
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter_f1 = 0
                        best_model_state = model.state_dict()
                    else:
                        patience_counter_f1 += 1
                else:
                    patience_counter_loss += 1
                    if patience_counter_loss >= early_stopping_patience:
                        # final check on f1 before stopping
                        y_prob_epoch = batched_inference(model, X_val_t, batch_size)
                        _, val_f1, _ = evaluations.find_optimal_threshold(y_val_t.cpu().numpy(), y_prob_epoch)
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_model_state = model.state_dict()
                            patience_counter_loss = 0
                            patience_counter_f1 = 0
                        else:
                            patience_counter_f1 += 1
                
                if patience_counter_loss >= early_stopping_patience and patience_counter_f1 >= early_stopping_patience:
                    break
                    
            except Exception as e:
                print(f"| Error in epoch {epoch+1}: {str(e)[:50]} |", end="", flush=True)
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return 1.0 # worst score
                continue

        print(f" {time.time() - start_time:.1f}s]", end="")

        if best_model_state:
            model.load_state_dict(best_model_state)
        
        y_prob = batched_inference(model, X_val_t, batch_size)
        _, best_f1, _ = evaluations.find_optimal_threshold(y_val_t.cpu().numpy(), y_prob)
        
        return 1.0 - float(best_f1)

    gc.collect()
    return obj


def retrain_and_evaluate(best_hp, X_train, y_train, X_test, y_test, batch_size, max_epochs=40, early_stopping_patience=8):
    """
    retrain best architecture on full training data, use test set for verification.
    """
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        best_hp = best_hp[2]

    device = select_device()
    print(f"Retraining on device: {device}")
    
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
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
        
    y_prob = batched_inference(model, X_test_t, batch_size)
        
    metrics = evaluations.classification_metrics(y_test, y_prob, threshold=0.5)
    best_thresh, best_f1, best_metrics = evaluations.find_optimal_threshold(y_test, y_prob)
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Threshold: {best_thresh:.4f}")
    print(f"  F1 Score:  {best_f1:.4f}")
    
    metrics.update({
        "optimal_threshold": best_thresh,
        "optimal_f1": best_f1,
        "optimal_precision": best_metrics['precision'],
        "optimal_recall": best_metrics['recall']
    })
    
    return model, metrics


def train_final_model(best_hp, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, max_epochs=50, early_stopping_patience=10, save_path=None):
    """
    train final model using best hyperparams.
    """
    print("="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        best_hp = best_hp[2]
        
    print(f"Hyperparameters: {best_hp}")
    
    device = select_device()
    print(f"Training on device: {device}")
    
    y_train_flat = y_train.flatten()
    y_val_flat = y_val.flatten()
    y_test_flat = y_test.flatten()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_flat, dtype=torch.float32).reshape(-1, 1).to(device)
    
    model = build_mlp_model(best_hp, X_train.shape[1], lr=0.001)
    model.to(device)
    
    num_pos = (y_train_flat == 1).sum()
    num_neg = (y_train_flat == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    print(f"Using positive class weight: {pos_weight:.4f}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    loss_fn = ManualBCELoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training for {max_epochs} epochs...")
    n_samples = len(X_train_t)
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_accum = 0.0
        batches = 0
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
            
        avg_train_loss = train_loss_accum / batches if batches else 0
        val_loss = batched_validation_loss(model, X_val_t, y_val_t, loss_fn, batch_size)
            
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

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Restored best model weight state.")
        
    if save_path:
        try:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")

    # final evaluation
    y_prob = batched_inference(model, X_test_t, batch_size)
    print("\nEvaluating final model on test set...")
    
    best_thresh, best_f1, best_metrics = evaluations.find_optimal_threshold(y_test_flat, y_prob)
    metrics_05 = evaluations.classification_metrics(y_test_flat, y_prob, threshold=0.5)
    
    metrics = {
        "optimal_threshold": best_thresh,
        "optimal_f1": best_f1,
        "optimal_precision": best_metrics['precision'],
        "optimal_recall": best_metrics['recall'],
        "optimal_roc_auc": best_metrics.get('roc_auc', None),
        "optimal_auprc": best_metrics.get('auprc', None),
        "threshold_05_f1": metrics_05['f1'],
        "threshold_05_precision": metrics_05['precision'],
        "threshold_05_recall": metrics_05['recall'],
        "threshold_05_roc_auc": metrics_05.get('roc_auc', None),
        "threshold_05_auprc": metrics_05.get('auprc', None),
    }
    
    print(f"\nResults @ Optimal Threshold ({best_thresh:.4f}):")
    print(f"  F1 Score:  {best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    
    return model, metrics


# ==================================================================================
# AUTOENCODER IMPLEMENTATION
# ==================================================================================

def build_ae_model(hyperparameters, input_shape):
    """builds autoencoder model."""
    device = select_device()
    
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
    
    # encoder
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
        
    # bottleneck
    layers.append(nn.Linear(in_dim, latent_dim))
    layers.append(nn.ReLU()) 
    
    # decoder
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
        
    # output (linear)
    layers.append(nn.Linear(in_dim, int(input_shape)))
    
    model = nn.Sequential(*layers)
    model.to(device, dtype=torch.float32)

    return model

def batched_reconstruction_loss(model, X_tensor, batch_size=1024):
    """calculate MSE reconstruction loss in batches."""
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
    """return squared error per sample."""
    model.eval()
    errors_list = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            recon = model(batch)
            batch_errors = torch.mean((batch - recon)**2, dim=1)
            errors_list.append(batch_errors)
            
    return torch.cat(errors_list).cpu().numpy()

def set_ae_optimizer_objective(X_train, y_train, X_val, y_val, max_epochs, batch_size, seed, early_stopping_patience, force_cpu=False):
    """autoencoder objective: minimize (1 - f1) via reconstruction error."""
    use_gpu = (not force_cpu) and (torch.cuda.is_available() or torch_directml.device() is not None)
    
    if use_gpu:
        try:
            device = select_device()
            if device.type == 'privateuseone':
                print("Warning: DirectML detected. If unstable, consider forcing CPU.")
            print(f"Optimizer using DEVICE: {device}")
        except:
            device = torch.device('cpu')
            print("Optimizer fell back to CPU.")
    else:
        device = torch.device('cpu')
        print("Optimizer using CPU.")
        
    # data prep
    y_val_np = np.array(y_val).flatten() if not isinstance(y_val, torch.Tensor) else y_val.cpu().numpy().flatten()
    X_val_np = np.array(X_val) if not isinstance(X_val, torch.Tensor) else X_val.cpu().numpy()
    
    total_val_limit = 20000 
    
    if len(X_val_np) > total_val_limit:
        print(f"Smart downsampling validation set to {total_val_limit}.")
        fraud_indices = np.where(y_val_np == 1)[0]
        normal_indices = np.where(y_val_np == 0)[0]
        
        n_fraud = len(fraud_indices)
        n_normal_target = total_val_limit - n_fraud
        if n_normal_target < 1000: n_normal_target = 1000
        
        np.random.seed(seed)
        if len(normal_indices) > n_normal_target:
            normal_indices = np.random.choice(normal_indices, size=n_normal_target, replace=False)
            
        final_indices = np.concatenate([fraud_indices, normal_indices])
        np.random.shuffle(final_indices)
        
        X_val_opt = X_val_np[final_indices]
        y_val_opt = y_val_np[final_indices]
    else:
        X_val_opt = X_val_np
        y_val_opt = y_val_np

    # dataloader
    X_train_tensor = torch.from_numpy(X_train).float()
    train_dataset = TensorDataset(X_train_tensor)
    
    X_val_t = torch.tensor(X_val_opt, dtype=torch.float32).to(device)
    
    input_noise_std = 0.01 
    
    def obj(vec):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        hp = hyp_optimizer.optimizer_vectors_to_ae_hyperparams(vec)
        
        try:
            model = build_ae_model(hp, X_train.shape[1])
            model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            use_pin = (device.type != 'cpu') and (device.type != 'privateuseone')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin, drop_last=True)
            
            best_val_loss = float('inf')
            patience = 0
            best_state = None
            
            for epoch in range(max_epochs):
                model.train()
                for batch_data in train_loader:
                    batch_x = batch_data[0].to(device, non_blocking=False)
                    
                    if input_noise_std > 0:
                        noisy_x = batch_x + torch.randn_like(batch_x) * input_noise_std
                    else:
                        noisy_x = batch_x
                    
                    optimizer.zero_grad()
                    recon = model(noisy_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()
                    
                val_loss = batched_reconstruction_loss(model, X_val_t, batch_size)
                print(".", end="", flush=True)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break
            
            if best_state:
                model.load_state_dict(best_state)
                
            with torch.no_grad():
                recon_errors = get_reconstruction_errors(model, X_val_t, batch_size)
            
            _, best_f1, _ = evaluations.find_optimal_threshold(y_val_opt, recon_errors)
            
            del model
            del optimizer
            return 1.0 - best_f1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| OOM |", end="")
                if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
                return 1.0
            else:
                print(f"Error: {e}")
                return 1.0
        
    return obj

def train_final_ae_model(best_hp, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, max_epochs=100, save_path=None):
    """
    train final ae on full data, evaluate on test.
    """
    print("="*60)
    print(f"FINAL AE TRAINING (Max Epochs: {max_epochs})")
    print("="*60)
    
    if isinstance(best_hp, (tuple, list)) and len(best_hp) >= 3:
        best_hp = best_hp[2]
    
    try:
        device = select_device()
    except:
        device = torch.device('cpu')
    
    X_train_tensor = torch.from_numpy(X_train).float()
    train_dataset = TensorDataset(X_train_tensor)
    
    use_pin = (device.type != 'cpu') and (device.type != 'privateuseone')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin, drop_last=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model = build_ae_model(best_hp, X_train.shape[1])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    patience_limit = 10
    
    input_noise_std = 0.01
    print(f"Training AE on {device} (Noise: {input_noise_std})...")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        
        for batch_data in train_loader:
            batch_x = batch_data[0].to(device, non_blocking=True)
            
            if input_noise_std > 0:
                noisy_x = batch_x + torch.randn_like(batch_x) * input_noise_std
            else:
                noisy_x = batch_x
            
            optimizer.zero_grad()
            recon = model(noisy_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
        avg_train_loss = train_loss / batches if batches else 0
        val_loss = batched_reconstruction_loss(model, X_val_t, batch_size)
        
        if epoch % 5 == 0:
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
        
    print("\nEvaluating AE Classification Performance on test set...")
    recon_errors = get_reconstruction_errors(model, X_test_t, batch_size)
    y_test_flat = np.array(y_test).flatten()
    
    thresh, f1, best_metrics = evaluations.find_optimal_threshold(y_test_flat, recon_errors)
    
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y_test_flat, recon_errors)
        auprc = average_precision_score(y_test_flat, recon_errors)
    except: 
        auc = None
        auprc = None
    
    metrics = {
        "optimal_threshold": thresh,
        "optimal_f1": f1,
        "optimal_precision": best_metrics['precision'],
        "optimal_recall": best_metrics['recall'],
        "optimal_roc_auc": auc,
        "optimal_auprc": auprc,
    }
    
    print(f"\nResults @ Optimal Threshold ({thresh:.6f}):")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    
    return model, metrics
