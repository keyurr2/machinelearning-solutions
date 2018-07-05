#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 13:12:09 2018

@author: keyur-r
"""

# Random Forest classifier with backward elimination


# Data Pre Processing - handling missing values, scaling features, categorical features 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backward_elimination import backwardElimination

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

# Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((4521,1)).astype(int), values = X, axis = 1)
SL = 0.05
X_opt = X[:, list(range(0,40))]
X_Modeled = backwardElimination(X_opt, SL, y)


# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Random Forest classifier to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 12, criterion = 'entropy', random_state = 0)
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
