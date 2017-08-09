#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def generateRandomPoints(nClusters=5, nPerCluster=100):
    points = []
    for n in range(nClusters):
        x = np.random.normal(loc=20*(np.random.random() - 0.5), scale=1.5, size=nPerCluster)
        y = np.random.normal(loc=20*(np.random.random() - 0.5), scale=1.5, size=nPerCluster)
        points.append(np.vstack([x,y]).T)
    points = np.vstack(points)
    return points

class Clusterer:
    def cluster(self, points, nCentroids=10):
        # shuffle data
        np.random.shuffle(points)
        # allocate data - 60% to training
        #                 20% to cross-validation
        #                 20% to test
        _train, _cv, _test = np.split(points, [int(points.shape[0] * 0.6), int(points.shape[0] * 0.8)])

        # for 25 times:
        #   randomly initialize centroids
        #   calculate cost and gradient
        #   while gradient > 0.001
        #     assign points to centroid
        #     assign centroid to mean of its assigned points
        #     calculate cost and gradient
        #   calculate cost on cross-validation set
        # return set of centroids with lowest cost

points = generateRandomPoints()
Clusterer().cluster(points)
plt.plot(points[:,0], points[:,1], 'bo')
plt.show()