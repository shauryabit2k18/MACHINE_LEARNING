# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:57:42 2019

@author: Shaurya Sinha
"""

#polynomial regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#splitting the dataset into train and test sets
"""from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train , Y_test = train_test_split(X , Y , test_size =  , random_state = 0) 
 """
 #we dont need to split because the dataset is already very samall and here we need very accurate results
 
 #fitting linear regression to the dataset
 from sklearn.linear_model import LinearRegression
 lin_reg = LinearRegression()
 lin_reg.fit(X , Y)
 
 #fitting polynomial regression to the dataset
 from sklearn.preprocessing import PolynomialFeatures
 poly_reg = PolynomialFeatures(degree = 4)
 X_poly = poly_reg.fit_transform(X)
 poly_reg.fit(X_poly , Y)
 lin_reg_2 = LinearRegression()
 lin_reg_2.fit(X_poly , Y)
 
 #visualising the linear regression results
 plt.scatter(X , Y , color = 'red')
 plt.plot(X , lin_reg.predict(X) , color = 'blue' )
 plt.title('TRUTH OR BLUFF[LINEAR REGRESSION]')
 plt.xlabel('position level')
 plt.ylabel('salary')
 plt.show()
 
 #visualising the polynomial regression results
 plt.scatter(X , Y , color = 'red')
 plt.plot(X , lin_reg_2.predict(poly_reg.fit_transform(X)) , color = 'blue' )
 plt.title('TRUTH OR BLUFF[POLYNOMIAL REGRESSION]')
 plt.xlabel('position level')
 plt.ylabel('salary')
 plt.show()
 
 #predicting a new result with linear regression
 lin_reg.predict([[6.5]])
 
 #predicting a new result with poynomial regression
 lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))