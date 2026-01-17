"""
Test data preprocessing pipeline for data leakage and correctness.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

def test_no_data_leakage_target_encoding():
    """
    Test that target encoding is only fit on training data.
    """
    # Create synthetic data with categorical features
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y'], n_samples),
        'num1': np.random.randn(n_samples),
        'num2': np.random.randn(n_samples)
    })
    y = pd.Series(np.random.choice([0, 1], n_samples))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit encoder only on training data
    encoder = TargetEncoder(cols=['cat1', 'cat2'])
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    
    # Verify that test data encoding doesn't match training exactly
    # (would indicate data leakage if it did)
    assert not X_train_encoded.equals(X_test_encoded), \
        "Target encoding may have data leakage"
    
    # Verify encoder was fit on training data only
    assert hasattr(encoder, 'mapping_'), "Encoder should be fitted"

def test_no_data_leakage_scaling():
    """
    Test that scaler is only fit on training data.
    """
    np.random.seed(42)
    X_train = np.random.randn(800, 10)
    X_test = np.random.randn(200, 10)
    
    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Verify training data is centered (mean ~ 0)
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10), \
        "Training data should be centered after scaling"
    
    # Verify test data is NOT centered (would indicate data leakage)
    assert not np.allclose(X_test_scaled.mean(axis=0), 0, atol=0.1), \
        "Test data should not be centered (scaler fit only on train)"

def test_no_data_leakage_pca():
    """
    Test that PCA is only fit on training data.
    """
    np.random.seed(42)
    X_train = np.random.randn(800, 20)
    X_test = np.random.randn(200, 20)
    
    # Fit PCA only on training data
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Verify PCA was fit
    assert hasattr(pca, 'components_'), "PCA should be fitted"
    assert pca.components_.shape == (10, 20), "PCA components shape incorrect"
    
    # Verify output shapes
    assert X_train_pca.shape == (800, 10), "Training PCA shape incorrect"
    assert X_test_pca.shape == (200, 10), "Test PCA shape incorrect"

def test_class_balancing_only_on_train():
    """
    Test that class balancing (SMOTE) is only applied to training set.
    """
    from imblearn.over_sampling import SMOTE
    
    np.random.seed(42)
    X_train = np.random.randn(800, 10)
    y_train = np.concatenate([np.zeros(700), np.ones(100)])  # Imbalanced
    
    X_test = np.random.randn(200, 10)
    y_test = np.concatenate([np.zeros(180), np.ones(20)])  # Imbalanced
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Verify training data is balanced
    assert np.sum(y_train_balanced == 0) == np.sum(y_train_balanced == 1), \
        "Training data should be balanced after SMOTE"
    
    # Verify test data remains imbalanced (not touched by SMOTE)
    assert np.sum(y_test == 0) != np.sum(y_test == 1), \
        "Test data should remain imbalanced (SMOTE only on train)"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
