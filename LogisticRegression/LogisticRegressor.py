#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:28:01 2018

@author: keyur-r
"""

# Logistic Regression
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

# Fitting logistic regression to training dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
TN, TP = cm[0,0], cm[1,1]
FN, FP = cm[1,0], cm[0,1]
accuracy = (TN + TP) / (TN + TP + FN + FP)
misclassification = (FN + FP) / (TN + TP + FN + FP)

# Visulization of data




