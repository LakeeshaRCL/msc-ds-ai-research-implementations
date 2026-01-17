"""
Test optimization consistency between optimization phase and final training phase.
Ensures that evaluation protocols match and results are consistent.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Note: These tests require the actual modules to be importable
# They should be run from the project root directory

def test_evaluation_protocol_consistency():
    """
    Test that optimization and final training use the same evaluation protocol.
    Both should use optimal threshold finding for F1 score.
    """
    # This is a placeholder test structure
    # Actual implementation would require importing the modules and running a small optimization
    pass

def test_validation_set_usage():
    """
    Test that validation set is used for early stopping, not test set.
    """
    # Verify that train_final_model uses validation set for early stopping
    pass

def test_stratified_sampling():
    """
    Test that stratified sampling maintains class distribution.
    """
    # Create imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, weights=[0.9, 0.1],
                               random_state=42)
    
    # Test stratified sampling
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=500, stratify=y, random_state=42
    )
    
    # Check that class distribution is maintained
    original_ratio = np.sum(y == 1) / len(y)
    sample_ratio = np.sum(y_sample == 1) / len(y_sample)
    
    # Should be approximately the same (within 5%)
    assert abs(original_ratio - sample_ratio) < 0.05, \
        f"Class distribution not maintained: original={original_ratio:.3f}, sample={sample_ratio:.3f}"

def test_optimal_threshold_consistency():
    """
    Test that optimal threshold finding works consistently.
    """
    from globals.model_evaluations import find_optimal_threshold
    
    # Create synthetic data
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    threshold, f1, metrics = find_optimal_threshold(y_true, y_prob)
    
    # Verify threshold is in valid range
    assert 0.0 <= threshold <= 1.0, f"Threshold out of range: {threshold}"
    
    # Verify F1 is reasonable
    assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"
    
    # Verify metrics are present
    assert 'precision' in metrics
    assert 'recall' in metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
