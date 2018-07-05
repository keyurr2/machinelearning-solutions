#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:15:07 2018

@author: keyur-r
"""

# Polynomial Linear Regression 

# Data Pre Processing - handling missing values, scaling features, categorical features 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Preparing features and labels from dataframe
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X, y)
# Visulizing results of Simple Linear 
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_simple.predict(X), color = "blue")
plt.title("Salary Prediction - Linear")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
# Fitting polynomial features to Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Visulizing results of polynomial
plt.scatter(X, y, color = "red")
# don't use X_poly in predict as it's already fit. So if X changes we don't need to run again.
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Salary Prediction - Polynomial")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()