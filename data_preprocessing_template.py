# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:27:57 2019

@author: Shaurya Sinha
"""

#simple linear regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#splitting the dataset into test and training dataset
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 1/3 , random_state = 0)

#feature scaling
"""from sklearn_preprocessing import StandardScalar
sc_X = StandardScalar()
X_train = sc_X.fit_transform(X_train)
sc_Y = StandardScalar()
Y_train = sc_X.fit_transform(Y_train)"""
#we could have used the above method for feature scaling the dataset but instead we will be
#using a different library of python in the next step that will take care of feature scaling all by itself

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

#predicting the test set results
Y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train , Y_train , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.title('SALARY VS EXPERIENCE[TRAINING SET]')
plt.xlabel('years of experience')
plt.ylable('salary')
plt.show()

#visualising the test set results
plt.scatter(X_test , Y_test , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.title('SALARY VS EXPERIENCE[TEST SET]')
plt.xlabel('years of experience')
plt.ylable('salary')
plt.show()