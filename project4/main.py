import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

from itertools import combinations
from functools import reduce

MAX_ITER = 20

def main():
    data = getData()
    kmeans(data, k=2)

### PART 1

def kmeans(data, k=2):
    """
    Implementation of the k means algorithm

    @param data: List of data points
    @param k: Number of clusters
    @returns: Classified cluster data points
    """
    count = 0
    centers = randCenters(data, k)
    oldCenters = None
    labels = []

    while not converged(oldCenters, centers, count):
        oldCenters = centers
        count += 1

        labels = getLabels(data, centers)
        centers = getCenters(data, labels, centers)
    print(centers)


def randCenters(data, k):
    """
    Initialize center points with random values

    @param data: List of data points
    @param k: Number of clusters
    @returns: List of randomized center points
    """
    return [data[np.random.randint(0, len(data), size=1)].flatten() for i in range(k)]


def converged(oldCenters, centers, count):
    """
    Checks if the center points have converged

    @param oldCenters: List of previous center points
    @param centers: Current center points
    @param count: Current number of iterations
    @returns: Bool
    """
    if oldCenters != None:
        return True if count >= MAX_ITER or np.array_equal(oldCenters, centers) else False

    return False

def getData():
    """
    Parses CSV for relevant data
    """
    return genfromtxt('data-1.txt', delimiter=',')


def getLabels(data, centers):
    """
    Assign data points to the k clusters based off of center points

    @param data: The list of data points
    @param centers: The list of center points
    @returns: List of labeled data points to clusters
    """
    labels = []

    for i, row in enumerate(data):
        prev = math.inf
        best = 0

        for cKey, center in enumerate(centers):
            cur = dist(row, center)

            if cur < prev:
                best = cKey
            prev = cur
        labels.append(best)

    return labels


def dist(row, center):
    """
    Calculates euclidean distance between two data points

    @param row: The data point
    @param center: The cluster center
    @returns Float representation of euclidean distance
    """
    return np.linalg.norm(row - center)


def getCenters(data, labels, centers):
    """
    Calculates updated center values for the k clusters

    @param data: The data set
    @param labels: Indicator for which cluster each row (point) sits at
    @param centers: List of center points
    @returns: An updated list of center points
    """
    sums = []
    newCenters = []

    for i, center in enumerate(centers):
        sums.append([np.zeros(data.shape[1]).flatten(), 0])

    for i, label in enumerate(labels):
        sums[label][0] = np.add(sums[label][0], data[i])
        sums[label][1] += 1

    for i, row in enumerate(sums):
        newCenters.append(np.divide(row[0], row[1]))

    return newCenters

### PART 2

def hacDistanceFunc(data, pair, distanceFunction):
    """
    Calculates the distance with a specific distance function

    @param data: List of points
    @param pair: A pair of clusters
    @param distanceFunction: A distance function
    @returns: Distance between two clusters with a specific distance function
    """
    a = np.array([data[i] for i in pair[0]])
    b = np.array([data[i] for i in pair[1]])
    return distanceFunction(a, b)


# https://elki-project.github.io/tutorial/hierarchical_clustering
def HAC(data, distanceFunc, threshold):
    """
    Hierarchical agglomerative clustering algorithm

    @param data: List of points
    @param distanceFunc: Distance function
    @param threshold: The threshold the tree is cut
    @returns: A clustering hierarchy
    """

    # Initialize necessary labels and each into into its own cluster
    labels = [0 for i in range(len(data))]
    clusters = {(i, ) for i in range(len(data))}

    j = len(clusters) + 1
    while True:
        clusterPairs = combinations(clusters, 2)

        # Calculate distance of each cluster pair in the form of (pair, distance)
        distanceScores = [(pair, hacDistanceFunc(data, pair, distanceFunc)) for pair in pairs]

        # Determine which pair is merged through best distance
        maximum = max(distanceScores, key=operator.itemgetter(1))

        # Stop if distnce below threshold
        if maximum[1] < threshold:
            break

        # Remove the pair that will be merged from cluster set, then merge/flatten them
        pair = maximum[0]
        clusters -= set(pair)
        flatPair = reduce(lambda x,y: x + y, pair)

        # Update labels for pair members
        for i in flatPair:
            labels[i] = j

        # Add new cluster to clusters
        clusters.add(flatPair)

        # End if no more clusters
        if len(clusters) == 1:
            break

        # Increment
        j += 1

    return labels


if __name__ == '__main__':
    main()
