#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:58:07 2018

@author: keyur-r
"""

# Multiple Linear Regression (Use allin template here)
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
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((53,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5,6]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

# So now we will take three features only into account.
X = X[:, [0,3,5]]
# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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

# Evaluating the Algorithm 
# http://canworksmart.com/using-mean-absolute-error-forecast-accuracy/
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error_val = mean_absolute_error(y_test, y_pred) # Mean Absolute Error
mean_squared_error_val = mean_squared_error(y_test, y_pred) # Mean Squared Error
root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_pred)) # Root Mean Squared Error 

