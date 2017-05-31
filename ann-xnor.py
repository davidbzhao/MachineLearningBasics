# David Zhao
# Last modified: 31 May 2017
# ann-xnor.py 
#
# Use a forward-propagation-only neural network structure
# to implement an xnor logic gate

import numpy as np
from math import exp

def sigmoid(x):
	return 1/(1+exp(-x))
sigmoid = np.vectorize(sigmoid)
# theta01 [2x3] - weights between 0 and 1
#  x0  x1  x2
# -30  20  20  <- AND
#  10 -20 -20  <- NOR
theta01 = np.array([-30,20,20,10,-20,-20]).reshape(2,3)

# theta12 [1x3] - weights between 1 and 2
#  x0  x1  x2
# -10  20  20
theta12 = np.array([-10,20,20])


# Assume user inputs valid data
print('first input (1 or 0)', end=' > ')
b0 = int(input())
print('second input (1 or 0)', end=' > ')
b1 = int(input())

a0 = np.array([1,b0,b1])
a1 = np.insert(sigmoid(theta01 @ a0), 0, 1)
a2 = sigmoid(theta12 @ a1)
print(round(a2.item(0)))
