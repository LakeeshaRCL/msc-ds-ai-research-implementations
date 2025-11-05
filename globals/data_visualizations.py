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
