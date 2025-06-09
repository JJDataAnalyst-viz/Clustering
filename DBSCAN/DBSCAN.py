import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import plotly.express as px

def blobs():
 return make_blobs(n_samples=300,centers=3,cluster_std=2,random_state=42)

def make_scatter(X,y):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

def make_DBSCAN(X,y):
    dbscan = DBSCAN(eps=1.7,n_jobs=-1).fit(X)
    labels = dbscan.labels_
    print(dbscan.components_)
    print(len(labels))
    print(dbscan.components_.shape)
    fig = px.scatter(x=X[:,0],y=X[:,1],color=labels)
    fig.show()

    
def main():
    X,y = blobs()
    make_scatter(X,y)
    make_DBSCAN(X,y)

if __name__ == "__main__":
  main()