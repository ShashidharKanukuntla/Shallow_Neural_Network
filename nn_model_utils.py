# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:57:59 2020

@author: Shashidhar
"""
import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(Y):
    return np.exp(Y)/sum(np.exp(Y)).reshape(1,Y.shape[1])

def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    # Uncomment below line for categorical data(softmax)
    # A2 = cross_entropy(A2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1] # number of example

    logprobs = np.multiply(Y,np.log(A2, where=A2>0))+np.multiply(1-Y,np.log(1-A2, where=(1-A2)>0))
    cost = (-1/m)*np.sum(logprobs)
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, X_test, Y_test, n_h, learning_rate, num_iterations = 10000, print_cost=False):
    
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    costs = []
    test_costs = []
    
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
         
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function
        cost = compute_cost(A2, Y, parameters)
        
        test_a2, _ = forward_propagation(X_test, parameters)
        
        test_cost = compute_cost(test_a2, Y_test, parameters)
 
        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads, learning_rate)
        
        costs.append(cost)
        test_costs.append(test_cost)
        
        # Print the cost
        if print_cost:
            print ("After iteration %i: Training cost = %f, test cost = %f" %(i, cost, test_cost))

    return parameters, costs, test_costs

def predict(parameters, X):
    
    A2, cache = forward_propagation(X, parameters)
    predictions = A2
    
    return predictions

def test_train_plot(training_costs, test_costs):
    training_costs = np.squeeze(training_costs)
    test_costs = np.squeeze(test_costs)
    plt.plot(training_costs, label='training cost')
    plt.plot(test_costs, label='testing cost')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title(" Training - Test costs")
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()