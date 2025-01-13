import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_arr(arr):
    sns.set_theme(style="whitegrid")

    # transpose since plotting feature / seaborn is being dumb
    arr = arr.T

    df = pd.DataFrame(arr)
    df_stack = df.stack().reset_index(name="correlation")

    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot(
        data=df_stack,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=2, aspect=1, sizes=(50, 100), size_norm=(-.2, .8),
    )

    plt.gca().invert_yaxis()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.gcf().set_size_inches(12, 6) 

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)