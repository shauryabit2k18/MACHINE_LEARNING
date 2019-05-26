# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:43:50 2019

@author: Shaurya Sinha
"""

# self organising mapos (SOM)

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
fraud = mappings[(6 , 2)]
fraud = sc.inverse_transform(fraud)