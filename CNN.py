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
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout 


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


#Configurating CNN model
cnn = Sequential() #import generic model
cnn.add(Conv1D(filters=1, kernel_size=4, strides=2, input_shape=(timesteps, 1))) #add 1D convulational layer
cnn.add(Flatten())
cnn.add(Dense(4, activation='selu')) #Regular BPNN layer   
cnn.add(Dense(1)) #Finish with 1 since we expect one output
cnn.compile(loss='mse') 

#Printing model summary
print(cnn.summary())

#Fit the model
cnn.fit(X_train, Y_train, epochs = 5, batch_size = 1, verbose=0)


#Prediction
Y_train_predict = cnn.predict(X_train)
Y_predict = cnn.predict(X_test)
#print(Y_test)
#print(Y_predict)
print("")
print("Test R2: {0}".format(metrics.r2_score(Y_test, Y_predict).round(2)))
print("")

#Plotting the results
real = np.concatenate((Y_train, Y_test), axis=0)
predict = np.concatenate((Y_train_predict, Y_predict), axis=0)
x_vline = len(data_X[0:threshold])

plt.plot(real, color='tab:blue', label = 'Y_Predict')
plt.plot(predict, color = 'tab:orange',  label = 'Y_test')
plt.vlines(x_vline,-2.1, 2.1, color='black')
plt.legend()
plt.show()