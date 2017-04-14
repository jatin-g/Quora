#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 05:07:08 2017

@author: Jatin
"""

import sklearn
from sklearn import datasets, linear_model
# Generate a dataset and plot it
import numpy as np
import matplotlib.pyplot as plt



np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


# Train the logistic rgeression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)
 


#clf.score tells the current accuracy of the training dataset(make moons)
clf.score(X,y)
y.mean()

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
    
## Plot the decision boundary
#plot_decision_boundary(lambda x: clf.predict(x))
#plt.title("Logistic Regression")

# Linear classifiers such as logisic regression are not able to fit the data,continued below
# means not able to separate the data perfectly.
#This is where neural network comes up. The hidden layer of neural network will learn the features

#Now we will create/train a 3 layer neural network with one input layer, one hidden layer and one
# output layer.
#We need to learn the paramters basically means finding the weights which reduces
# the error rate
#We are using gradient descent method to find the minimum error.

#As input gradient descent requires gradients of the loss function with respect to 
# to our parameters. use backpropogation for that.



#Defining some variables for gradient descent

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (randomly chosen)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
#layerSize=3


#Now defining loss function y-yhat
#\begin{aligned}  L(y,\hat{y}) = - \frac{1}{N} \sum_{n \in N} \sum_{i \in C} y_{n,i} \log\hat{y}_{n,i}  \end{aligned}  

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'],model['W2'], model['b2']
    #Forward propagation to calculate our predictions, means y hat
    z1= X.dot(W1) + b1   #z1 is the input to hidden layer
    a1 = np.tanh(z1)     #output from hidden layer after applying activation function(tanh)
    z2 = a1.dot(W2) + b2   #input to output layer
    exp_scores = np.exp(z2)
    
    #Use of keepdims-True = axes which are reduced are kept in the result as dimensions with size one.
    #here we are reducing axes=1 which is rows 
    
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True) #axis=1 is sum along rows
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
    
    
    
    # Helper function to predict an output (0 or 1)
    #Here we are doing forward propogation and returning the class with highest probability
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
    
    
#Here comes the function to train Neural Network. Implementing gradient descent using back propagation 
#Here we will find the parameters(weights)
# nn_hdim = number of nodes in hidden layer, num_passes = Number of passes through training data for gradient descent
# print_loss = if true, print loss after every 1000 iterations. 
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    #Initialize the parameters W1, b1, W2, b2 to random values. We need to find these only
    
    #randn(2,4) = 2 arrays for size 4. 
    
    #Basically in W1 we have 2 nodes in first layer(nn_input_dim) and 
    #3 nodes in hidden layer(nn_hdim) so we are creating inputs from 2 first layer nodes for each of the 3 hidden layer 
    #nodes using randn
    
    #np.zeros(1,nn_hdim=3) means one array of size 3
    
    #W2 = np.random.randn(nn_hdim, nn_output_dim) means 3,2 so 3 arrays of size two (as output layer has 2 nodes)
    
    #q1 why dividing by sqaure root
     
    
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_output_dim)
    b2 = np.zeros((1, nn_output_dim))
    
    print(W1,b1,W2,b2)
    #Model is what we return in the end
    model = {}

#Gradient descent. For each batch
    for i in range(0, num_passes):
        #Forward propagation
        #.dot is the inner product which is the scalar function of two vectors, basically equal to the product of their magnitudes
        z1= X.dot(W1) + b1   #z1 is the input to hidden layer
        a1 = np.tanh(z1)     #output from hidden layer after applying activation function(tanh)
        z2 = a1.dot(W2) + b2   #input to output layer
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True) #axis=1 is sum along rows
       
#        print("z1 a1 z2 and probs are", z1, a1, z2, probs)
        
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    
    return model
    

# Build a model with a 3-dimensional hidden layer

model = build_model(3, print_loss=True)
#print("z1 a1 z2 and probs are", z1, a1, z2, probs)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
    
    
    
    
    
    