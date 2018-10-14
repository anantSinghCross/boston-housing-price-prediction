import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
data = pd.read_csv("boston.csv", delimiter = ',')
dataset = np.asarray(data)
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
#print (dataset)
