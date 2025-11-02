import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

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

def get_categorical_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def show_dataframe_head(df, n=5, columns=None, reset_index=False):
    """
    Display the first n rows of a DataFrame with optional column selection.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to display
    n : int, default=5
        Number of rows to display from the top
    columns : list, str, or None, default=None
        Specific columns to display. Can be:
        - None: Show all columns
        - list: List of column names ['col1', 'col2']
        - str: Single column name 'col1'
    reset_index : bool, default=False
        Whether to reset the index in the display

    Returns:
    --------
    pandas.DataFrame
        The head of the DataFrame (for chaining operations)

    Examples:
    ---------
    >>> show_dataframe_head(df)  # Show first 5 rows, all columns
    >>> show_dataframe_head(df, n=10)  # Show first 10 rows
    >>> show_dataframe_head(df, columns=['Name', 'Age'])  # Specific columns
    >>> show_dataframe_head(df, n=3, columns='Name')  # Single column
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return None

    # Handle column selection
    if columns is not None:
        # Convert single column string to list
        if isinstance(columns, str):
            columns = [columns]

        # Validate columns exist
        invalid_cols = [col for col in columns if col not in df.columns]
        if invalid_cols:
            print(f"Warning: Columns not found: {invalid_cols}")
            columns = [col for col in columns if col in df.columns]

        if not columns:
            print("No valid columns to display")
            return None

        result = df[columns].head(n)
    else:
        result = df.head(n)

    # Reset index if requested
    if reset_index:
        result = result.reset_index(drop=True)

    print(result)
    return result


def get_categorical_value_counts(dataframe:pd.DataFrame, categorical_columns:list)-> pd.DataFrame:
    """
    Generate a new DataFrame with categorical column names and their value counts.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The original DataFrame
    categorical_columns : list
        List of categorical column names to analyze

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns ['column', 'value_count']
    """
    result_data = []

    for col in categorical_columns:
        if col in dataframe.columns:
            value_count = dataframe[col].nunique()
            result_data.append({'column': col, 'value_count': value_count})

    return pd.DataFrame(result_data)

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    One-hot encode a specified categorical column in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame
    column : str
        The name of the categorical column to encode

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with one-hot encoded columns added and the original column removed
    """

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))

    # Concatenate the original DataFrame (excluding the original column) with the encoded DataFrame
    result_df = pd.concat([df.drop(columns=columns), encoded_df], axis=1)

    return result_df

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame
    columns : list
        List of column names to drop

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the specified columns removed
    """
    return df.drop(columns=columns, axis=1)