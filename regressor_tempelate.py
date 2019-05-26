# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:07:32 2019

@author: Shaurya Sinha
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the Regression model to the dataset
#create ur regressor here

# Predicting a new result 
Y_pred = regressor.predict([[6.5]])


# Visualising the Polynomial Regression results(for higher resolution and smoother curve)
X_grid = np.arrange(min(X) , max(X) , 0.1)
X_grid = Xgrid.reshape((len(X_grid) , 1))
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()