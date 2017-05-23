import numpy as np
import math
import operator
from itertools import combinations
from functools import reduce
from shared import dist

def part2(data):
    print('\nStarting Problem 2.1')
    singleHAC(data)

    print('\nStarting Problem 2.2')
    completeHAC(data)


def singleHAC(data):
    """
    Single Hierarchical agglomerative clustering algorithm

    @param data: List of points
    @returns: A clustering hierarchy
    """

    # Initialize necessary variables
    clusters = []
    distances = np.zeros((data.shape[0], data.shape[0]))
    distances.fill(np.inf)

    # Each object into its own cluster
    for i in range(data.shape[0]):
        clusters.append([i])

    # Calculate the distances between each cluster
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distances[i][j] = dist(data[i], data[j])

    # Repeat until only one cluster remaining
    while distances.shape[1] > 1:
        ci = 0
        cj = 0

        # Obtain minimum distance pair clusters
        indexes = distances.argmin()
        dimensions = distances.shape
        ci, cj = np.unravel_index(indexes, dimensions)
        distance = np.min(distances)
        distances[ci, cj] = np.inf

        # Create new distances without old union clusters
        newDistances = distances
        newDistances = np.delete(newDistances, [ci, cj], axis=0)
        newDistances = np.delete(newDistances, [ci, cj], axis=1)

        # Add single union cluster
        unionCluster = np.zeros(newDistances.shape[1])
        newDistances = np.vstack([newDistances, unionCluster])

        # Compute the new distance between the union cluster to the other clusters
        for i in range(newDistances.shape[1]):
            newDistances[-1][i] = min(distances[ci][i], distances[cj][i])

        distances = newDistances

        # Get 10 clusters
        if distances.shape[1] < 11:
            print("Cluster A: {0} | Cluster B: {1} | Height: {2} | Distance: {3}".format(ci, cj, distances.shape[1], distance))


def completeHAC(data):
    """
    Complete Hierarchical agglomerative clustering algorithm

    @param data: List of points
    @returns: A clustering hierarchy
    """

    # Initialize necessary variables
    clusters = []
    distances = np.zeros((data.shape[0], data.shape[0]))
    # distances.fill(np.inf)

    # Each object into its own cluster
    for i in range(data.shape[0]):
        clusters.append([i])

    # Calculate the distances between each cluster
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distances[i][j] = dist(data[i], data[j])

    # Repeat until only one cluster remaining
    while distances.shape[1] > 1:
        ci = 0
        cj = 0

        # Obtain maximum distance pair clusters
        indexes = distances.argmax()
        dimensions = distances.shape
        ci, cj = np.unravel_index(indexes, dimensions)
        distance = np.max(distances)
        distances[ci, cj] = 0

        # Create new distances without old union clusters
        newDistances = distances
        newDistances = np.delete(newDistances, [ci, cj], axis=0)
        newDistances = np.delete(newDistances, [ci, cj], axis=1)

        # Add single union cluster
        unionCluster = np.zeros(newDistances.shape[1])
        newDistances = np.vstack([newDistances, unionCluster])

        # Compute the new distance between the union cluster to the other clusters
        for i in range(newDistances.shape[1]):
            newDistances[-1][i] = max(distances[ci][i], distances[cj][i])

        distances = newDistances

        # Get 10 clusters
        if distances.shape[1] < 11:
            print("Cluster A: {0} | Cluster B: {1} | Height: {2} | Distance: {3}".format(ci, cj, distances.shape[1], distance))
