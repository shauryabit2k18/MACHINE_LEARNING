# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:25:29 2019

@author: Shaurya Sinha
"""

#hierarchical clustering

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3 , 4]].values

#using the dendrogram to find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.title('dendrogram')
plt.ylabel('euclidian distance')
plt.xlabel('customers')
plt.show()

#fitting the hirarchical clustering to the mall datast
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean' , linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the cluster
plt.scatter(X[y_hc == 0 , 0] , X[y_hc == 0 , 1], s = 100 , c = 'red' , label = 'Careful')
plt.scatter(X[y_hc == 1 , 0] , X[y_hc == 1 , 1], s = 100 , c = 'blue' , label = 'standard')
plt.scatter(X[y_hc == 2 , 0] , X[y_hc == 2 , 1], s = 100 , c = 'green' , label = 'target')
plt.scatter(X[y_hc == 3 , 0] , X[y_hc == 3 , 1], s = 100 , c = 'cyan' , label = 'Careless')
plt.scatter(X[y_hc == 4 , 0] , X[y_hc == 4 , 1], s = 100 , c = 'magenta' , label = 'sensible')
plt.title('cluster of clients')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100')
plt.legend()
plt.show()