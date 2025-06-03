import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans,kmeans_plusplus
from sklearn.decomposition import PCA

def load_iris() -> pd.DataFrame:
    '''
    Load the Iris dataset from scikit-learn and return it as a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the Iris dataset with features and target.

    '''
    iris = sns.load_dataset("iris")
    return iris

def iris_pca(iris:pd.DataFrame) -> np.array:
    pass



if __name__ == "__main__":
    load_iris()
