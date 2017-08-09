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
        centroids = _train[:nCentroids]
        prev_cost = -1
        #   assign points to centroid
        #   calculate cost and gradient
        buckets, centroids, cost = self.bucketAndUpdate(_train, centroids)
        print(cost)
        #   while gradient > 0.001
        while prev_cost != cost:
            prev_cost = cost
        #     assign points to centroid
        #     assign centroid to mean of its assigned points
        #     calculate cost and gradient
            buckets, centroids, cost = self.bucketAndUpdate(_train, centroids)
            print(cost)
        #   calculate cost on cross-validation set
        # return set of centroids with lowest cost
        return centroids
        
    def bucketAndUpdate(self, _train, centroids):
        cost = 0
        buckets = [[] for n in range(centroids.shape[0])]
        for n in range(_train.shape[0]):
            costs = np.sum((_train[n] - centroids) ** 2, axis=1)
            buckets[np.argmin(costs)].append(_train[n])
            cost += np.min(costs)
        for n in range(len(buckets)):
            centroids[n] = np.average(np.vstack(buckets[n]), axis=0)
        return buckets, centroids, cost


points = generateRandomPoints()
clusterer = Clusterer()
centroids = clusterer.cluster(points)
buckets, tmp, points = clusterer.bucketAndUpdate(points, centroids)
for n in range(len(buckets)):
    bucket_points = np.vstack(buckets[n])
    plt.plot(bucket_points[:,0], bucket_points[:,1], 'o')
plt.show()
