#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:23:41 2018

@author: keyur-r
"""

# Logistic Regression 

# Data Pre Processing - handling missing values, scaling features, categorical features 

# Loading dataset using library
import statsmodels.api as sm
dataset = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dataset['affair'] = (dataset.affairs > 0).astype(int)

# Preparing features and labels from dataframe
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling of train and test data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # fit and transform
X_test = sc_X.transform(X_test) # already data is fit so only transform here

# Fitting Logistic Regression to Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Prediction of Test set results
y_pred = classifier.predict(X_test)

# Test result accuracy
train_accuracy = classifier.score(X_test, y_pred)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#



