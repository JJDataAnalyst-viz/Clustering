import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles

def make_circle():
    data = make_circles(1000)
    return data


X,y = make_circle()

