from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot2d_dataset(X, y, title: str, legend):
    df = pd.DataFrame(
        data = {"x1": X[:, 0], "x2": X[:, 1], "y": y}
    )
    df["y"].astype("category")
    fig, ax = plt.subplots()
    ax = sns.scatterplot(data = df, x = "x1", y = "x2", hue = "y", s = 40,
                         palette = {-1: "black", 
                                    0: "darkred", 
                                    1: "darkblue", 
                                    2: "darkgreen",
                                    3: "darkorange",})
    legend_handles, _= ax.get_legend_handles_labels()
    ax.set_title(label = title)
    ax.legend(legend_handles, legend)