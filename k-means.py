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

class Clusterer:
    def cluster(self, x, y, nCentroids=10):
        # allocate data - 60% to training
        #                 20% to cross-validation
        #                 20% to test
        # for 25 times:
        #   randomly initialize centroids
        #   calculate cost and gradient
        #   while gradient > 0.001
        #     assign points to centroid
        #     assign centroid to mean of its assigned points
        #     calculate cost and gradient
        #   calculate cost on cross-validation set
        # return set of centroids with lowest cost

x, y = generateRandomPoints()
plt.plot(x, y, 'bo')
plt.show()