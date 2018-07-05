#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:17:47 2018

@author: keyur-r
"""


# Reading from CSV

# Importing libraries 
import numpy as np # for numerical analysis
import matplotlib.pyplot as plt # visulization
import pandas as pd # dataframe

# Reading from csv file and preparing Panda Dataframe
dataset = pd.read_csv('Data.csv')

# In case of numeric information
# dataset.shape # (500,11) -> 500 rows and 11 colums
# dataset.info() -> info for each column
# dataset.describe() -> for each column, count, mean, std, min, max
# dataset.head(5) -> first five data information
 
# Extracting values from Dataframe with iloc

# all data from dataset data.iloc[<row selection>, <column selection>]
data_all = dataset.iloc[:, :].values

data_1 = dataset.iloc[0].values # first row
data_2 = dataset.iloc[:, 0].values # first column, will return panda series
data_3 = dataset.iloc[[0], [0]].values # first row and first column
data_4 = dataset.iloc[:, :-1].values # all rows with excluding last column
data_5 = dataset.iloc[[0,1,2], [0,1,2]] # first, second, third rows and columns

# Selection of data using named index
dataset.set_index("last_name", inplace=True)
dataset.head()
data_6 = dataset.loc['Andrade']

# Select rows with index values 'Andrade' and 'Veness', with all columns between 'address' and 'county'
data_7 = dataset.loc[['Andrade', 'Veness'], 'address':'county']
# Select same rows, with just 'first_name', 'address' and 'city' columns
data_8 = dataset.loc['Andrade':'Veness', ['first_name', 'address', 'city']]
 
# Change the index to be based on the 'id' column
data_9 = dataset.set_index('id', inplace=True)
# select the row with 'id' = 487
data_10 = dataset.loc[487]
