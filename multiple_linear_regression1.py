# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:26:35 2019

@author: Shaurya Sinha
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#encoading the categorical variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:, 1:]
#here the fist ie 0th indexed column has been excluded from X to avoid the dummy variable X

#splitting the dataset into train test sets
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)

#fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

#predicting the test set results
Y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1)
#here ve added X0 in the starting of the X array to calculate b0*x0
X_opt = X[:, [0 , 1 , 2 , 3 , 4 , 5]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
#we see that the P value is the highest for x3 so we will remove it
X_opt = X[:, [0 , 1 , 2 , 4 , 5]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
#we see that the P value is the highest for x2 so we will remove it
X_opt = X[:, [0 , 1 , 4 , 5]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
#we see that the P value is the highest for x3 so we will remove it
X_opt = X[:, [0 , 1 , 4 ]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
