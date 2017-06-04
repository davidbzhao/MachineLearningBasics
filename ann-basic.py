# David Zhao
# Last modified 4 June 2017
# ann-basic.py
#
# Implement a functional neural network

import numpy as np
from random import random

L = 3 # number of layers
dim = [2,2,1] # size of each layer
theta = [] # weights

# Initialize random weights for each layer
# An additional bias node is prepended to the first layer in each pair of layers 
def initWeights():
	for cnt in range(L-1):
		theta.append([[20*(random()-0.5)]*(dim[cnt]+1)]*dim[cnt+1])

def main():
	initWeights()
	print(theta)
	# forwardPropOne()
	# backwardProp()
	# forwardPropOne()
main()