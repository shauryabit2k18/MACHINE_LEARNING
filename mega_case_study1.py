# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:29:01 2019

@author: Shaurya Sinha
"""

# mega case study

# part 1 - identify the frauds with self organising maps (SOM)

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv ')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0 , 1))
X = sc.fit_transform(X)

# training the SOM
from minisom import MiniSom
som = MiniSom(x = 10 , y = 10 , input_len = 15 , sigma = 1.0 , learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X , num_iteration = 100)

#visualising the results
from pylab import bone , pcolor , colorbar , plot , show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o' , 's']
colors = ['r' , 'g']
for i, x in enumerate(X):
	w = som.winner(x)
	plot(w[0] + 0.5 , w[1] + 0.5 , markers[y[i]] , markeredgecolor = colors[y[i]] , markerfacecolor = 'None' , markersize = 10 , markeredgewidth = 2)
show()

# finding the frauds
mappings = som.win_map(X)
fraud = np.concatenate(( mappings[(8 , 1)] , mappings[(8 , 8)]) , axis = 0)
fraud = sc.inverse_transform(fraud)

# part 2 - going from the unsuperwised to superwised deep learning

# creating the matrix of features
customers = dataset.iloc[:, 1:].values

# creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
	if dataset.iloc[i , 0] in fraud:
		is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers , is_fraud , batch_size = 1 , epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the probabilities of fraud
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1] , y_pred) , axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]