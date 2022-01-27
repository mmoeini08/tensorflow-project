# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:47:15 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Deep learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import GridSearchCV 
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf



#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'elec_nov2017_60607'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#How much time before event do we want to include?
timesteps = int(input("Timesteps:"))

#Keeping values only to make it simpler to define X and Y
data = data.values

#Defining X and Y
data_Y = data[timesteps:len(data)+1]
data_X = []
for i in range(0, len(data)-timesteps):
    data_X.append(data[i:(i+timesteps)])


#Train test split, but careful keeping last 20% as testing
#data_train, data_test = train_test_split(data, train_size = 0.8) #Only use stratify for classification
threshold = int(0.8 * len(data_X))
X_train = data_X[0:threshold]
Y_train = data_Y[0:threshold]
X_test = data_X[threshold:len(data)+1]
Y_test = data_Y[threshold:len(data)+1]


#Standard Scaling only on X should be ok since X and Y have the same data
Xmean, Xstd = np.mean(X_train), np.std(X_train)
X_train = (X_train - Xmean) / Xstd
X_test = (X_test - Xmean) / Xstd
Y_train= (Y_train - Xmean) / Xstd
Y_test = (Y_test - Xmean) / Xstd
'''
#MinMax
Xmin, Xmax = np.min(X_train), np.max(X_train)
X_train = (X_train - Xmin) / (Xmax - Xmin)
X_test = (X_test - Xmin) / (Xmax - Xmin)
Y_train = (Y_train - Xmin) / (Xmax - Xmin)
Y_test = (Y_test - Xmin) / (Xmax - Xmin)
'''

def create_gru(unit, activation):
    gru = tf.keras.models.Sequential()
    gru.add(Dropout(0.2))
    gru.add(Dense(3))#, activation='relu')
    gru.add(GRU(units=unit, activation=activation, recurrent_activation=activation))
    gru.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
    return gru
# gru = KerasClassifier(build_fn=create_gru, epochs=5, batch_size=1, verbose=0)
gru=KerasClassifier(build_fn=create_gru)
# params={'units':units}
params={'activation':["relu", "exponential", "sigmoid"], 
        'unit':[5, 10, 15],'batch_size':[10, 20, 50]}
gs=GridSearchCV(estimator=gru, param_grid=params, cv=5)
gs.fit(X_train, Y_train, epochs = 5, verbose=0)

best_params=gs.best_params_
accuracy=gs.best_score_
print(best_params)
print(accuracy)