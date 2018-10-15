import numpy as np
import pickle
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
print(dataset.shape)
print(X.shape)
print(Y.shape)

'''     Now we will define three different Keras Models that will be different 
        with respect to 'number of hidden layers', 'number of epochs' and
        'nuerons in the initial layer'.
'''
# define base model
def baseline_model():
	# create model (this is the main structure of what the model will be)
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model (this tells how the model is to be trained)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 7
np.random.seed(seed)
# evaluate model with standardised dataset
estimator = KerasRegressor(build_fn = baseline_model , epochs = 100 , batch_size = 5 , verbose = 1)

# evaluating model using the kfold method
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))

print("FITTING THE MODEL NOW")
# fitting the model
estimator.fit(X,Y,batch_size = 5 , epochs = 100)
# pickling the model
pickle.dump(estimator,open("model.pkl","wb"))
print("MODEL HAS BEEN PICKLED SUCCESSFULLY")
