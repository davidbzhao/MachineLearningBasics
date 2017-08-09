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
        np.random.shuffle(points)
        # allocate data - 60% to training
        #                 20% to cross-validation
        #                 20% to test
        _train, _cv, _test = np.split(points, [int(points.shape[0] * 0.6), int(points.shape[0] * 0.8)])
        best_cost = float('inf')
        best_centroids = _train[:nCentroids]
        for n in range(25):
            np.random.shuffle(_train)
            centroids = _train[:nCentroids]
            prev_cost = -1
            buckets, centroids, cost = self.bucketAndUpdate(_train, centroids)
            print(cost)
            while prev_cost != cost:
                prev_cost = cost
                buckets, centroids, cost = self.bucketAndUpdate(_train, centroids)
                # print(cost)
            cv_cost = self.cost(_cv, centroids)
            if cv_cost < best_cost:
                best_cost = cv_cost
                best_centroids = centroids
        test_cost = self.cost(_test, best_centroids)
        return best_centroids, test_cost
        
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

    def cost(self, points, centroids):
        cost = 0
        for n in range(points.shape[0]):
            costs = np.sum((points[n] - centroids) ** 2, axis=1)
            cost += np.min(costs)
        return cost


points = generateRandomPoints()
clusterer = Clusterer()
centroids, test_cost = clusterer.cluster(points)
print(test_cost)
buckets, tmp, points = clusterer.bucketAndUpdate(points, centroids)
for n in range(len(buckets)):
    bucket_points = np.vstack(buckets[n])
    plt.plot(bucket_points[:,0], bucket_points[:,1], 'o')
plt.show()
