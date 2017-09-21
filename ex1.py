# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:53:54 2017

@author: Vsevold Loik
"""

import numpy as np
import os
import matplotlib.pyplot as plt

path = 'C:\\Users\\Nikita Loik\\Documents\\Andrew Ng - Machine Learning\\Assignments'
os.chdir(path)

filename = 'ex1data1.txt'
data = np.loadtxt(filename, delimiter = ',')
# to insert a column of ones into x use np.insert. We use data[:, :-1] to select all but last column
# and to preserve 2-demsnionality with ':-1' (compared to '-1' which returns 1-demensional array)
X = np.insert(data[:, :-1], 0, 1, axis = 1)
y = data[:, -1:]
sampleSize, numThetas = X.shape

# Visualize the data
plt.scatter(X[:,1],y, c= 'red', marker = 'x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profits by population')
plt.show()

## Gradient Descent
# Initilize fitting parameters
theta = np.zeros((numThetas,1))
iterations = 1500
alpha = 0.01


#def computeCost(X,y,theta):
#    '''
#    COMPUTECOST Compute cost for linear regression
#    COMPUTECOST(X, y, theta) computes the cost of using theta as the
#    parameter for linear regression to fit the data points in X and y
#    '''
#    #Initialize some useful values
#    m = len(y) # number of training examples
#    mysum = 0
#    
#    for i in range(m):
#        mysum += (theta[0]*X[i,0] + theta[1]*X[i,1] - y[i])**2
#    J = 1/(2*m) * mysum
#    return(J)
#    
#def gradientDescent(X,y,theta,alpha,iterations):
#    '''
#    GRADIENTDESCENT Performs gradient descent to learn theta
#    Updates theta by taking num_iters gradient steps with learning rate alpha
#    '''
#    #Initialize some useful values
#    m = len(y) # number of training examples
#    J_history = []
#    mysum0 = 0
#    mysum1 = 0
#    for j in range(iterations):
#        for i in range(m):
#            mysum0 += (theta[0]*X[i,0] + theta[1]*X[i,1] - y[i])
#            mysum1 += (theta[0]*X[i,0] + theta[1]*X[i,1] - y[i])*X[i,1]
#        #print('gradient:' + str(1/m * mysum0) + ',' + str(1/m * mysum1))
#        theta[0] -= alpha * 1/m * mysum0
#        theta[1] -= alpha * 1/m * mysum1
#        
#        J_history + = computeCost(X,y,theta))
#        print('theta = ' + str(theta) + '. Cost = ' + str(float(computeCost(X,y,theta))))
#    return J_history
    
    
def computeCost(X,y,theta):
    '''
    Compute cost for linear regression using theta as the
    parameter for linear regression to fit the data points in X and y
    '''
    J = np.sum((np.dot(X, theta) - y)**2)/(2*sampleSize)
    return J
    
def gradientDescent(X,y,theta,alpha,iterations):
    '''
    Performs gradient descent to learn theta
    Updates theta by taking num_iters gradient steps with learning rate alpha
    '''
    J_history = []
    for i in range(iterations):
        theta = theta - alpha * 1/sampleSize * np.dot(X.T, (np.dot(X, theta) - y))
        J_history += [computeCost(X,y,theta)]
    return theta, J_history
    
    