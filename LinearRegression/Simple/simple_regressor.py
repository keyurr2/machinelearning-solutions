#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:07:24 2018

@author: keyur-r
"""

# Simple Linear Regression (Use data_preprocessor template here)
# Data Pre Processing - handling missing values, scaling features, categorical features 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('slr01.csv')

# Preparing features and labels from dataframe
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data -> 
# if three string category then replaced with 0,1,2 
#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Now we don't want that these three category 0,1,2 means like 0 < 1 < 2
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

# Encoding dependent variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)
# Note : No need this for y as there are only two values 0 and 1

# Splitting the dataset into Training and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling of train and test data
# Not needed as sklearn.linear_model will take care it self.
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train) # fit and transform
#X_test = sc_X.transform(X_test) # already data is fit so only transform here
# no need to do feature scaling for y as there are only two values (yes/no)


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

# Visulization of results

# First plot predictions for training set and compare with ground truth
plt.scatter(X_train, y_train, color = 'red') # Ground truth values for train
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Predicted values for train
plt.title("Best Prices vs List Prices - Training")
plt.xlabel("List Prices")
plt.ylabel("Best Prices")

# Second plot predictions for test set and compare with ground truth
plt.scatter(X_test, y_test, color = 'red') # Ground truth values for train
# Here we don't need to use x_test as model (Linear line) is already fitted.
plt.plot(X_train, regressor.predict(X_train), color = 'black') 
plt.title("Best Prices vs List Prices - Testing")
plt.xlabel("List Prices")
plt.ylabel("Best Prices")

# Evaluating the Algorithm 
# http://canworksmart.com/using-mean-absolute-error-forecast-accuracy/
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error = mean_absolute_error(y_test, y_pred) # Mean Absolute Error
mean_squared_error = mean_squared_error(y_test, y_pred) # Mean Squared Error
root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_pred)) # Root Mean Squared Error 

# you can go through with other dataset 
# get your hands dirty and predict the result.