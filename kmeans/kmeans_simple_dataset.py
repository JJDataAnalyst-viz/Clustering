import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.cluster import kmeans_plusplus,KMeans



def make_circle():
    data = make_circles(1000,noise=0.05)
    return data

def plot_circles(X,y):
    palette = sns.dark_palette((20, 60, 50), input="husl")
    sns.set_style("darkgrid", {"grid.color": ".1", "grid.linestyle": ":"})
    sns.scatterplot(x = X[:,0],y = X[:,1],hue= y,palette=palette)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(["0","1"])
    plt.title("Circles for KMeans")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_facecolor("#AFAFAF")
    plt.savefig("kmeans_comparison_images/circle_true_labels.svg")
    plt.show()

   

if __name__ == "__main__":
    X,y = make_circle()
    plot_circles(X,y)

