import sys
import numpy as np
import csv
import math
import operator
from numpy import genfromtxt
from itertools import combinations
from functools import reduce

MAX_ITER = 1000

def main():
    fName = 'data-1.txt'
    try:
        fName=sys.argv[1]
    except Exception as e:
        print('No data file provided, defaulting to {0}'.format(fName))
    data = getData(fName)

    print('\nStarting Problem 1.1')
    # SSEs, labels, centers, iterations = kmeans(data, k=4)

    print('\nStarting Problem 1.2')

    print('\nStarting Problem 2.1')
    # kmeans(data, k=2)
    print(data[0:2])
    HAC(data[0:5], 0, 100)

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
    SSEs = [[] for i in range(k)]

    print('Calculating k-means for k={0}'.format(k))
    while not converged(oldCenters, centers, count):
        oldCenters = centers
        count += 1
        labels = getLabels(data, centers)
        centers = getCenters(data, labels, centers)

        calcSSE(labels, centers, SSEs, data, k)

    return SSEs, labels, centers, count
    print('Converged after {0} iteration(s)'.format(count))


def calcSSE(labels, centers, SSEs, data, k):
    values = [[] for i in range(k)]

    for row, label in enumerate(labels):
        values[label].append(data[row])

    for i, value in enumerate(values):
        SSEs[i].append(np.sum((values[i] - centers[i])**2))


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

def getData(fName='data-1.txt'):
    """
    Parses CSV for relevant data
    """
    print('Loading data...')
    return genfromtxt(fName, delimiter=',')


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
    for i in range(len(centers)):
        if labels.count(i) == 0:
            centers[i] = np.random.randint(0, len(data), size=1)

    for i, center in enumerate(centers):
        sums.append([np.zeros(data.shape[1]).flatten(), 0])

    for i, label in enumerate(labels):
        sums[label][0] = np.add(sums[label][0], data[i])
        sums[label][1] += 1

    for i, row in enumerate(sums):
        newCenters.append(np.divide(row[0], row[1]))

    return newCenters

### PART 2
def singleLinkDistance(clusterA, clusterB):
    """
    Compute distance between two clusters through single link

    @param clusterA: List of points for cluster A
    @param clusterB: List of points for cluster B
    @returns: Single link Euclidean distance
    """
    # allPairs = {}
    # for i in clusterA:
    #     for j in clusterB:
    #         if (i, j) not in allPairs:
    #             allPairs[(i,j)] = dist(i, j)
    # return min(allPairs)
    allPairs = []
    for i in clusterA:
        for j in clusterB:
            allPairs.append(dist(i, j))
    return min(allPairs)

def completeLinkDistance(clusterA, clusterB):
    """
    Compute distance between two clusters through complete link

    @param clusterA: List of points for cluster A
    @param clusterB: List of points for cluster B
    @returns: Complete link Euclidean distance
    """
    allPairs = []
    for i in clusterA:
        for j in clusterB:
            allPairs.append(dist(i, j))
    return max(allPairs)

def hacDistanceFunc(data, pair, distanceFunction):
    """
    Calculates the distance with a specific distance function

    @param data: List of points
    @param pair: A pair of clusters
    @param distanceFunction: A distance function (0 for single, 1 for complete)
    @returns: Distance between two clusters with a specific distance function
    """
    a = np.array([data[i] for i in pair[0]])
    b = np.array([data[i] for i in pair[1]])
    if distanceFunction == 0:
        return singleLinkDistance(a, b)
    else:
        return completeLinkDistance(a, b)


# https://elki-project.github.io/tutorial/hierarchical_clustering
def HAC(data, distanceFunc, threshold):
    """
    Hierarchical agglomerative clustering algorithm

    @param data: List of points
    @param distanceFunc: Distance function (0 for single, 1 for complete)
    @param threshold: The threshold the tree is cut
    @returns: A clustering hierarchy
    """

    # Initialize necessary labels and each object into its own cluster
    labels = [0 for i in range(len(data))]
    clusters = {(i, ) for i in range(len(data))}
    print(clusters)
    j = len(clusters) + 1
    distanceScores = {}

    while True:
        clusterPairs = combinations(clusters, 2)
        # print("clusterpairs")
        # for x in clusterPairs:
        #    print(x)

        # Calculate distance of each cluster pair in the form of (pair, distance)

        # Evaluate distance between pairs, memo existing
        for pair in clusterPairs:
            if pair not in distanceScores or False:
                distanceScores[pair] = hacDistanceFunc(data, pair, distanceFunc)
        # distanceScores = [(pair, hacDistanceFunc(data, pair, distanceFunc)) for pair in clusterPairs]
        print("distance score")
        print(distanceScores)
        # Determine which pair is merged through best distance
        maximum = max(distanceScores, key=operator.itemgetter(1))
        print("max:")
        print(maximum)
        # Stop if distance below threshold
        if distanceScores[maximum] < threshold:
            break

        # Remove the pair that will be merged from cluster set, then merge/flatten them
        pair = maximum

        print("pair")
        print(pair)

        clusters -= set(pair)
        flatPair = reduce(lambda x,y: x + y, pair)

        print("flatPair:")
        print(flatPair)

        # Update labels for pair members
        for i in flatPair:
            labels[i] = j

        # Add new cluster to clusters
        clusters.add(flatPair)

        print("cluster lens")
        print(len(clusters))

        # End if no more clusters
        if len(clusters) == 1:
            break

        print("LABELS")
        print(labels)
        print("INTERATION")
        print(j)
        # Increment
        j += 1

    print(labels)
    return labels


if __name__ == '__main__':
    main()
