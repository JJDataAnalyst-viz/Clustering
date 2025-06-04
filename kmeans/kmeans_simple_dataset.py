import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.cluster import kmeans_plusplus,KMeans

def make_circle() -> pd.DataFrame:
    '''
    Load toy dataset form scikit-learn library 

    Returns:
        pd.DataFrame: return X and y values from make_circle module
    '''
    data = make_circles(1000,noise=0.05)
    return data

def plot_circles(X: np.array,y: np.array) -> None:
    # palette = sns.dark_palette((20, 60, 50), input="husl")
    sns.set_style("darkgrid", {"grid.color": ".1", "grid.linestyle": ":"})
    sns.scatterplot(x = X[:,0],y = X[:,1],hue= y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Circles for KMeans")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_facecolor("#AFAFAF")
    plt.savefig("kmeans_comparison_images/circle_true_labels.svg")
    plt.show()

def create_centroids(X: np.array,y: np.array):
    centroids, indices = kmeans_plusplus(X,n_clusters=2)
    print(centroids)
    print(indices)

def kmeans_algorithm(X):
    kmeans = KMeans(n_clusters=2).fit_predict(X)
    return kmeans
if __name__ == "__main__":
    X,y = make_circle()
    # create_centroids(X,y)
    # circle = kmeans_algorithm(X)
    plot_circles(X,y)

