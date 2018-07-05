#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:26:03 2018

@author: keyur-r
"""


# Data Pre Processing - handling missing values, scaling features, categorical features 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('bank.csv')

# Finding unrelated column index to remove from dataset
dataset.drop('contact', axis=1, inplace=True)
dataset.drop('default', axis=1, inplace=True)

# Replace negative value with 0
dataset['pdays'] = dataset['pdays'].replace([-1], 0)
dataset['housing'].replace(('yes', 'no'), (1, 0), inplace=True)
dataset['loan'].replace(('yes', 'no'), (1, 0), inplace=True)
dataset['y'].replace(('yes', 'no'), (1, 0), inplace=True)

# Splitting features and results
y = dataset.iloc[:, dataset.columns.get_loc("y")].values
dataset.drop('y', axis=1, inplace=True)

dataset = pd.get_dummies(dataset, columns=['job','marital', 'education', 'month', 'poutcome'], drop_first=True)
X = dataset.iloc[:, :].values


# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Gaussian Naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
TN, TP = cm[0,0], cm[1,1]
FN, FP = cm[1,0], cm[0,1]
accuracy = (TN + TP) / (TN + TP + FN + FP)
# score = classifier.score(X_test,y_test) # alternative of above one
misclassification = (FN + FP) / (TN + TP + FN + FP)

# Visulization of data
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01) \
#                      , np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# for i,j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
#                 c = ListedColormap(('red','green'))(i), label = j)
# plt.title("Logistic Regression - Training set")
# plt.xlabel("Age")
# plt.ylabel("Balance")
