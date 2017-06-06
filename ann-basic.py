# David Zhao
# Last modified 4 June 2017
# ann-basic.py
#
# Implement a functional neural network

import numpy as np
from random import random, sample
from math import exp

dim = [2,2,2,1] # size of each layer
L = len(dim) # number of layers
theta = [] # weights
lam = 0.01 # regularization factor 

# Initialize random weights for each layer
# An additional bias node is prepended to the first layer in each pair of layers
# 	If you had layers of dimensions 4, 2, and 1, the weights would be initialized
#	as arrays of dimensions as such, [[2x5],[1x3]].
def initWeights():
	for cnt in range(L-1):
		theta.append((np.random.rand(dim[cnt+1], dim[cnt]+1)-0.5)*5)

# Apply a sigmoid function to an input z,
# vectorized to work on numpy structures
def sigmoid(z):
	return 1/(1+exp(-z))
sigmoid = np.vectorize(sigmoid)

# Forward propagate with x, an input point
def forwardPropOne(x):
	x = np.array(x)
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
		sens.insert(0, np.multiply(np.dot(theta[cnt].T[1:],sens[0]), np.multiply(a[cnt],1-a[cnt])))
	grad = []
	for cnt in range(L-1):
		grad.append(np.dot(sens[cnt],np.insert(a[cnt],0,1,axis=0).T))
	return grad

def train(x, y):
	grad = [0*t for t in theta]
	reg = theta
	m = len(y)
	for i in range(m):
		a = forwardPropOne([[x[0][i]], [x[1][i]]])
		g = backwardProp(a,y[i])
		grad = [grad[k] + g[k] for k in range(L-1)]
	for i in range(L-1):
		reg[i].T[0] = 0
		theta[i] = theta[i] - 0.03*(grad[i]/m + lam*reg[i])
		# print(forwardPropOne(x))
	return

def main():
	initWeights()

	# x,y = [[0,0,1,1],[0,1,0,1]],[1,0,0,1]
	x = [[n/1000 for n in sample(range(10000),100)], [n/1000 for n in sample(range(10000),100)]]
	y = [((x[0][i] > 0.5) + (x[1][i] > 0.5)) % 2 == 0 for i in range(len(x[0]))]
	# x,y = [[0,0,1,1],[0,1,0,1]],[0,0,0,1]
	print(forwardPropOne([[0.75],[0.25]]))
	print('theta',theta)
	train(x,y)
	print(forwardPropOne([[0.75],[0.25]]))
	print('theta',theta)
main()