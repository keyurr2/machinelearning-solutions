#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:02:19 2018

@author: keyur-r
"""

# Multiple Linear Regression (Use backward_elimination template here)
#The data (X1, X2, X3, X4, X5, X6) are by city.
#X1 = death rate per 1000 residents
#X2 = doctor availability per 100,000 residents
#X3 = hospital availability per 100,000 residents
#X4 = annual per capita income in thousands of dollars
#X5 = State
#X6 = population density people per square mile
#we will predict population density people per square mile

# Data Pre Processing - handling missing values, scaling features, categorical features 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('mlr07.csv')

# Preparing features and labels from dataframe
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

# Encoding categorical data ->  
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# Now we don't want that these three category 0,1,2 means like 0 < 1 < 2
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap 
# If three categories than three colums with values like 0 , 0 , 1 
# features like california, florida, newyork
# So instead of three we can use two colums like if 0,0 then it's newyork
# if 1,0 then it's california
# thus we can eliminate one column of dummy variables
# it won't affect our information
X = X[:, 1:]

# Now we will perform backward elimination for feature selection.
# Steps :
#Select significant level to stay in model (SL = 0.05)
#Fit model with all features
#Check p-values for all features and select one feature having highest p-value.
#If P > SL then remove that feature and fit model again and perform the same. 
#(Remove one by one only).
#If no any features found with P > SL then our model is ready.

# Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((53,1)).astype(int), values = X, axis = 1)

# method 1 : Backward Elimination with p-values only
import statsmodels.formula.api as sm
def backwardEliminationP(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
# method 2 : Backward Elimination with p-values and Adjusted R Squared
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((53,7)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
X_Modeled = backwardElimination(X_opt, SL)


# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Feature Scaling of train and test data
# Not needed as sklearn.linear_model will take care it self.

# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# To retrieve the intercept
r_intercept = regressor.intercept_
# For retrieving the slope (coefficient of x)
r_coef = regressor.coef_


# Predicting the test set result.
y_pred = regressor.predict(X_test)

