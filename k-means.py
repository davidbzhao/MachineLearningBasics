#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def generateRandomPoints(nClusters=5, nPerCluster=100):
    x = []
    y = []
    for n in range(nClusters):
        x.append(np.random.normal(loc=20*(np.random.random() - 0.5), scale=1.5, size=nPerCluster))
        y.append(np.random.normal(loc=20*(np.random.random() - 0.5), scale=1.5, size=nPerCluster))
    return np.hstack(x), np.hstack(y)

x, y = generateRandomPoints()
plt.plot(x, y, 'bo')
plt.show()