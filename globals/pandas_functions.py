import pandas as pd
import os

# supportive functions
def dataset_dimension(name: str, dataset: pd.DataFrame):
    print(f"{name} dataset dimension: {dataset.shape}")

def get_null_count_df(dataset: pd.DataFrame) -> pd.DataFrame:
    null_counts = dataset.isnull().sum()
    null_df = null_counts.reset_index()
    null_df.columns = ["feature", "count"]
    return null_df

def get_null_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = get_null_count_df(df)
    null_counts["missing_percentage"] = (null_counts["count"] / len(df)) * 100
    return null_counts

def get_null_values_by_threshold_range(df: pd.DataFrame, lower_bound: float, upper_bound:float) -> pd.DataFrame:
    null_summary = get_null_value_summary(df)
    filtered_nulls = null_summary[
        (null_summary["missing_percentage"] > lower_bound) &
        (null_summary["missing_percentage"] <= upper_bound)
    ]

    return filtered_nulls

def export_dataframe_to_csv(df, directory, filename):
    # validate DataFrame
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    # ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'

    # create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # construct full file path
    file_path = os.path.join(directory, filename)

    # export DataFrame to CSV
    df.to_csv(file_path, index=False)

    print(f"DataFrame successfully exported to: {file_path}")

    return file_path