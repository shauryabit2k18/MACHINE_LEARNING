# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:13:18 2019

@author: Shaurya Sinha
"""

# artificial nural networ
# installing theano
# installing tenserflow
# installing keras

# part 1 - data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#avoiding the dummy variable trap for the country column
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part 2 - now lets make a ann:-
# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initializing the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))
classifier.add(Dropout(p = 0.1))

# adding the second hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' ))
classifier.add(Dropout(p = 0.1))

# adding the output layer
classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# fitting the ANN to the training dataset
classifier.fit(X_train , y_train , batch_size = 10 , epochs = 100)

# part 3 - making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#predicting a new observation
new_prediction = classifier.predict(sc.transform(np.array([[0.0 , 0 , 600 , 1 , 40 , 3 , 60000 , 2 , 1 , 1 , 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

# part 4 - evaluating , improving and tuning the ANN

# evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))
	classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' ))
	classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
	return classifier
classifier = KerasClassifier(build_fn = build_classifier , batch_size = 10 , epochs = 100)
accuracies = cross_val_score(estimator = classifier , X = X_train , y = y_train , cv = 10 , n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# improving the ANN
# droupout regularization to reduce overfitting if needed

# tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimiser):
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))
	classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' ))
	classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid'))
	classifier.compile(optimizer = optimiser , loss = 'binary_crossentropy' , metrics = ['accuracy'])
	return classifier
classifier = KerasClassifier(build_fn = build_classifier )
parameters = {'batch_size' : [25 , 32], 
	      'nb_epoch' : [100 , 500],
	      'optimiser' : ['adam' , 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
			   param_grid = parameters,
			   scoring = 'accuracy',
			   cv = 10)
grid_search.fit(X_train , y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_






