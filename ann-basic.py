# David Zhao
# Last modified 4 June 2017
# ann-basic.py
#
# Implement a functional neural network

import numpy as np
from random import random
from math import exp

L = 3 # number of layers
dim = [2,2,1] # size of each layer
theta = [] # weights

# Initialize random weights for each layer
# An additional bias node is prepended to the first layer in each pair of layers
# 	If you had layers of dimensions 4, 2, and 1, the weights would be initialized
#	as arrays of dimensions as such, [[2x5],[1x3]].
def initWeights():
	for cnt in range(L-1):
		theta.append((np.random.rand(dim[cnt+1], dim[cnt]+1)-0.5)*20)

# Apply a sigmoid function to an input z,
# vectorized to work on numpy structures
def sigmoid(z):
	return 1/(1+exp(-z))
sigmoid = np.vectorize(sigmoid)

# Forward propagate with x, an input point
def forwardPropOne(x):
	x = np.matrix(x)
	a = [x]
	for cnt in range(L-1):
		x = sigmoid(np.dot(theta[cnt],np.insert(x,0,1,axis=0)))
		a.append(x)
	return a

# Backward propagate to update weights given
# the forward propagation activation values, a,
# and the expected result, y
def backwardProp(a, y):
	global theta
	y = np.array(y)
	sens = [a[-1]-y] # sensitivities of the cost function to each pre-activation value
					 # initialized with the last layer sensitivity of output - expected
	for cnt in range(L-2, 0, -1):
		sens.insert(0, np.multiply(theta[cnt].T[1:]*sens[0], np.multiply(a[cnt],1-a[cnt])))

	grad = []
	for cnt in range(L-1):
		grad.append(np.dot(sens[cnt],np.insert(a[cnt],0,1,axis=0).T))
	theta = [theta[i] - grad[i] for i in range(len(theta))]

def main():
	initWeights()

	x,y = [[0,0,1,1],[0,1,0,1]],[1,0,0,1]
	a = forwardPropOne(x)
	print(a)
	backwardProp(a, y)
	a = forwardPropOne(x)
	print(a)
main()