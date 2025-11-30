import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data visualization functions
def show_bar_plot(data: pd.DataFrame, x_label: str, y_label:str, title, hue, figure_size=(12,6), color_palette="pastel"):
    plt.figure(figsize=figure_size)
    plt.title(title)
    sns.barplot(data=data, x=x_label, y=y_label, hue=hue, palette=color_palette)
    plt.show()

def show_histogram(
    data: pd.DataFrame,
    column: str,
    title: str = "",
    bins: int = 20,
    hue: str = None,
    figure_size=(12, 6),
    color_palette="pastel"
):
    plt.figure(figsize=figure_size)
    plt.title(title)
    sns.histplot(
        data=data,
        x=column,
        hue=hue,
        bins=bins,
        palette=color_palette,
        kde=True
    )
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()



def show_boxplot(data, column, title="", figure_size=(8, 5), color="skyblue", horizontal=False):

    plt.figure(figsize=figure_size)
    plt.title(title)
    if horizontal:
        sns.boxplot(x=data[column], color=color)
        plt.xlabel(column)
    else:
        sns.boxplot(y=data[column], color=color)
        plt.ylabel(column)
    plt.show()


def plot_categorical_distribution(df, column):

    value_counts = df[column].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values, palette='Set2', hue=value_counts.index)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

    # Add count labels
    for i, v in enumerate(value_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_pca(explained_variance, cumulative_variance):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Variance by Component')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_class_count_plot(data, column, title=None, figure_size=(10, 6), color_palette="Set2", show_percentages=True):
    """
    Display a count plot for class distribution in classification datasets.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series or array-like
        The data containing class labels. Can be a DataFrame, Series, or numpy array.
    column : str or None
        Column name if data is a DataFrame. Use None if data is a Series or array.
    title : str, optional
        Plot title. If None, auto-generates title based on column name.
    figure_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    color_palette : str, default="Set2"
        Seaborn color palette name.
    show_percentages : bool, default=True
        Whether to show percentages alongside counts.
    
    Returns:
    --------
    None (displays plot)
    
    Examples:
    ---------
    # With DataFrame
    show_class_count_plot(y_train_df, 'isFraud')
    
    # With Series or array
    show_class_count_plot(y_train.to_numpy().ravel(), None, title='Class Distribution')
    """
    import numpy as np
    
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column parameter is required when data is a DataFrame")
        values = data[column]
        col_name = column
    elif isinstance(data, pd.Series):
        values = data
        col_name = data.name if data.name else "Class"
    else:
        # Assume numpy array or list
        values = pd.Series(data)
        col_name = column if column else "Class"
    
    # Get value counts
    value_counts = values.value_counts().sort_index()
    total = len(values)
    
    # Create figure
    plt.figure(figsize=figure_size)
    
    # Create count plot
    ax = sns.countplot(x=values, palette=color_palette, order=value_counts.index)
    
    # Set title
    if title is None:
        title = f'Class Distribution: {col_name}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set labels
    plt.xlabel(col_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count and percentage labels on bars
    for i, (idx, count) in enumerate(value_counts.items()):
        percentage = (count / total) * 100
        
        if show_percentages:
            label = f'{count:,}\n({percentage:.2f}%)'
        else:
            label = f'{count:,}'
        
        ax.text(i, count, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add summary statistics in text box
    stats_text = f'Total: {total:,}\n'
    stats_text += f'Classes: {len(value_counts)}\n'
    if len(value_counts) == 2:
        # Binary classification - show imbalance ratio
        class_0, class_1 = value_counts.iloc[0], value_counts.iloc[1]
        ratio = max(class_0, class_1) / min(class_0, class_1)
        stats_text += f'Imbalance Ratio: {ratio:.2f}:1'
    
    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

