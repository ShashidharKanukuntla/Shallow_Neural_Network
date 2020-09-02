# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:18:09 2020

@author: Shashidhar
"""
import numpy as np

# X should be of shape (no. of features, no. of examples)
def normalizeData(x):
    return x/np.max(x, axis=1, keepdims=True)

# Y should be of shape (1, no. of examples)
def getCategoricaldata(Y):
    Y_cat = np.zeros((np.max(Y)+1,Y.shape[1]), dtype='int64')
    for i in range(Y.shape[1]):
        Y_cat[Y[0,i],i]=1
    return Y_cat

# X should be of shape (no. of features, no. of examples), Y should be of shape (1, no. of examples) 
def test_train_split(X, Y, ratio):
    X_train = X[:, :int(X.shape[1]*(1-ratio))]
    X_test = X[:, int(X.shape[1]*(1-ratio)):]
    Y_train = Y[:,:int(Y.shape[1]*(1-ratio))]
    Y_test = Y[:,int(Y.shape[1]*(1-ratio)):]
    return X_train, X_test, Y_train, Y_test