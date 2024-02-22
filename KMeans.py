'''

Name: Bharath Radhakrishnan

Part 1 - Implementation of KMeans Clustering Algorithm + Optional Section

'''
import math
import numpy as np
import random

class KMeans:

    def __init__(self, k=5, max_iterations=100, balanced=False):
        """
        Initializes the KMeans clustering algorithm with specified parameters.

        Parameters:
        - k: Number of clusters
        - max_iterations: Maximum number of iterations for convergence
        - balanced: If True, balance the clustures
        """
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced

    def fit(self, X):
        """
        Fits the KMeans algorithm to the input data.

        Parameters:
        - X: List of data points

        Returns:
        - instance_cluster: List containing cluster assignments for each data point
        - centroids: List of final cluster centroids
        """
        n = len(X)
        if n < self.k:
            print("Error!")

        instance_cluster = []
        centroids = []
        max_cluster_size = (n*1.2)/self.k

        for i in range(n):
            instance_cluster.append(-1)

        for i in range(self.k):
            centroids.append(X[random.randint(0, n-1)])

        for iter in range(self.max_iterations):
            # Assign clusters
            cluster_sizes = [0 for i in range(self.k)]
            for j in range(n):
                dist = []
                for k in range(self.k):
                    dist.append(self.euclidean_dist(X[j], centroids[k]))
                closest_centroid = dist.index(min(dist))
                # sorted_centroids = sorted(centroids, key=lambda x: self.euclidean_dist(X[j], x))
                if not self.balanced:
                    instance_cluster[j] = closest_centroid
                    cluster_sizes[closest_centroid] += 1
                else:
                    if instance_cluster.count(closest_centroid) < max_cluster_size:
                        instance_cluster[j] = closest_centroid
                    else:
                        instance_cluster[j] = random.choice([i for i in range(self.k) if cluster_sizes[i] < max_cluster_size])

            # Update centroids
            new_centroids = []
            for l in range(self.k):
                sum_l = [0 for i in range(len(X[0]))]
                count_l = 0
                for m in range(n):
                    if instance_cluster[m] == l:
                        count_l += 1
                        for i in range(len(X[0])):
                            sum_l[i] += X[m][i]
                if count_l != 0:
                    for i in range(len(X[0])):
                        sum_l[i] /= count_l
                    new_centroids.append(sum_l)
                elif self.balanced:
                    min_count = min([cluster_sizes[c] for c in range(self.k)])
                    candidate_centroids = [i for i, count in enumerate(instance_cluster) if count == min_count]
                    new_centroids.append(X[random.choice(candidate_centroids)])
                else:
                    new_centroids.append(centroids[l])
            centroids = new_centroids
        return instance_cluster, centroids
    
    # Euclidean distance between two vectors
    @staticmethod
    def euclidean_dist(v1, v2):
        square_sum = 0
        for i in range(len(v1)):
            square_sum += (v1[i] - v2[i])**2
        return math.sqrt(square_sum)
    
if __name__ == "__main__":
    kmeans = KMeans(k=2)
    X = [ [0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10] ]
    print(kmeans.fit(X))