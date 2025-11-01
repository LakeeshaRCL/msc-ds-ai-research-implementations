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


