"""
Helper functions for consistent validation and evaluation.
"""
import numpy as np
from sklearn.model_selection import train_test_split

def get_stratified_sample(X, y, sample_size, random_state=42):
    """
    Get a stratified sample maintaining class balance.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    sample_size : int
        Desired sample size
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_sample, y_sample : tuple
        Stratified sample of features and labels
    """
    if sample_size >= len(X):
        print(f"Sample size ({sample_size}) >= dataset size ({len(X)}). Using full dataset.")
        return X, y
    
    # Use stratified sampling to maintain class balance
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=sample_size,
        stratify=y,
        random_state=random_state
    )
    return X_sample, y_sample

def verify_class_distribution(original_y, sampled_y, tolerance=0.05):
    """
    Verify that class distribution is maintained after sampling.
    
    Parameters:
    -----------
    original_y : array-like
        Original labels
    sampled_y : array-like
        Sampled labels
    tolerance : float
        Maximum allowed difference in class ratio
        
    Returns:
    --------
    bool : True if distribution is maintained within tolerance
    """
    if len(np.unique(original_y)) != 2:
        return True  # Not binary classification
    
    original_ratio = np.sum(original_y == 1) / len(original_y)
    sampled_ratio = np.sum(sampled_y == 1) / len(sampled_y)
    
    return abs(original_ratio - sampled_ratio) < tolerance

def check_data_leakage(X_train, X_val, X_test, feature_names=None):
    """
    Basic check for data leakage by comparing feature statistics.
    
    Parameters:
    -----------
    X_train, X_val, X_test : array-like
        Training, validation, and test sets
    feature_names : list, optional
        Names of features for reporting
        
    Returns:
    --------
    dict : Dictionary with leakage check results
    """
    results = {
        'leakage_detected': False,
        'warnings': []
    }
    
    # Convert to numpy if needed
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    
    # Check if validation/test means are too close to training means
    # (would indicate scaling was done before split)
    train_means = X_train.mean(axis=0)
    val_means = X_val.mean(axis=0)
    test_means = X_test.mean(axis=0)
    
    # If validation/test are centered (mean ~ 0) but training is not,
    # this might indicate leakage
    val_centered = np.allclose(val_means, 0, atol=0.01)
    test_centered = np.allclose(test_means, 0, atol=0.01)
    train_centered = np.allclose(train_means, 0, atol=0.01)
    
    if (val_centered or test_centered) and not train_centered:
        results['leakage_detected'] = True
        results['warnings'].append(
            "Validation/test sets appear centered but training set is not. "
            "Possible scaling leakage."
        )
    
    return results
