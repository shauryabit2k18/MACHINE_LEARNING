# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:54:51 2019

@author: Shaurya Sinha
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('labeledTrainData.tsv', delimiter="\t" , quoting = 3)
dataset = dataset.drop(['id'], axis=1)
dataset.head()


df2 = pd.read_csv('testData.tsv', delimiter="\t" , quoting = 3)
df2 = dataset.drop(['id'], axis=1)
df2.head()

df1 = pd.read_csv('imdb_master.csv' , encoding = "latin-1")
df1.head()

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0 , 25000):
        review = re.sub('[^a-zA-Z]' , ' ' , dataset['review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        while 'br' in review:
	        review.remove("br")
        review = ' '.join(review)
        corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 40000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting random forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 500 , criterion = 'entropy' , random_state= 0)
classifier.fit(X_train , y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix     #this is a function nat a class
cm = confusion_matrix(y_test , y_pred)
