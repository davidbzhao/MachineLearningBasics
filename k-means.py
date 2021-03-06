#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def generateRandomPoints(nClusters=5, nPerCluster=100):
    points = []
    for n in range(nClusters):
        x = np.random.normal(loc=20*(np.random.random() - 0.5), size=nPerCluster)
        y = np.random.normal(loc=20*(np.random.random() - 0.5), size=nPerCluster)
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
            while prev_cost != cost:
                prev_cost = cost
                buckets, centroids, cost = self.bucketAndUpdate(_train, centroids)
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
        n = 0
        while n < len(buckets):
            if len(buckets[n]) == 0:
                del buckets[n]
                np.delete(centroids, n, axis=0)
            else:
                centroids[n] = np.average(np.vstack(buckets[n]), axis=0)
                n += 1
        return buckets, centroids, cost

    def cost(self, points, centroids):
        cost = 0
        for n in range(points.shape[0]):
            costs = np.sum((points[n] - centroids) ** 2, axis=1)
            cost += np.min(costs)
        return cost

def main():
    points = generateRandomPoints()
    clusterer = Clusterer()
    centroids, test_cost = clusterer.cluster(points, 5)
    print(centroids.shape)
    print(test_cost)
    buckets, tmp, points = clusterer.bucketAndUpdate(points, centroids)
    for n in range(len(buckets)):
        bucket_points = np.vstack(buckets[n])
        plt.scatter(bucket_points[:,0], bucket_points[:,1], s=10)
    plt.title('K-Means Clustering')
    plt.text(0, 0, 'Error : ' + str(test_cost))
    plt.plot(centroids[:,0], centroids[:,1], 'k^')
    plt.axis('equal')
    plt.show()
main()
