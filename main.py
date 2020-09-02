# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:40:29 2020

@author: Shashidhar
"""
# Importing the libraries
import numpy as np
import pandas as pd
from preprocess_utils import normalizeData, test_train_split, getCategoricaldata
from nn_model_utils import test_train_plot, nn_model, predict

np.seterr(divide = 'ignore') # To ignore divide by zero

# Importing the dataset
dataset_train = pd.read_csv('train.csv')

X = dataset_train.iloc[:, 1:].values.T
Y = dataset_train.iloc[:, 0].values.T
Y = Y.reshape((1,Y.shape[0]))


X_list = []
Y_list = []


for i in range(Y.shape[1]):
    if(Y[0,i])==0 or (Y[0,i])==1:
        X_list.append(True)
        Y_list.append(True)
    else:
        X_list.append(False)
        Y_list.append(False)
        
X_fil = X[:,X_list]
Y_fil = Y[0,Y_list].reshape(1, X_fil.shape[1])

# Uncomment below line for categorical data(softmax)        
# Y_cat= getCategoricaldata(Y)

# Uncomment below line for non-image data
# X_Norm = normalizeData(X)

X_Norm = X_fil/255.0 # Comment this line for non-image data

X_train, X_test, Y_train, Y_test = test_train_split(X_Norm, Y_fil, 0.2)

parameters, costs, test_costs = nn_model(X_train, Y_train, X_test, Y_test, n_h = 192, learning_rate=0.25, num_iterations = 1000, print_cost=True)

Y_prediction_train = predict(parameters, X_train)
Y_prediction_test = predict(parameters, X_test)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

test_train_plot(costs, test_costs)
