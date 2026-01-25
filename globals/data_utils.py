
from sklearn.model_selection import train_test_split
def show_class_distribution(x_df, y_df, title="Class Distribution"):
    print(f"\n{title}:")
    print(f"  Total samples: {len(x_df)}")
    print(f"  Y df samples: ", y_df)
    print(f"  Class 0 (non-fraud): {(y_df == 0).sum()} ({(y_df == 0).sum() / len(y_df) * 100:.2f}%)")
    print(f"  Class 1 (fraud): {(y_df == 1).sum()} ({(y_df == 1).sum() / len(y_df) * 100:.2f}%)")


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