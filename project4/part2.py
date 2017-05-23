import numpy as np
import math
import operator
from itertools import combinations
from functools import reduce

from shared import dist

def part2(data, distanceFunc, threshold):
    print('\nStarting Problem 2.1')
    HAC(data, distanceFunc, threshold)


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
        print("distance score\n{0}".format(distanceScores))
        # Determine which pair is merged through best distance
        maximum = max(distanceScores, key=operator.itemgetter(1))
        print("max\n{0}".format(maximum))
        # Stop if distance below threshold
        if distanceScores[maximum] < threshold:
            break

        # Remove the pair that will be merged from cluster set, then merge/flatten them
        pair = maximum

        print("pair\n{0}".format(pair))

        clusters -= set(pair)
        flatPair = reduce(lambda x,y: x + y, pair)

        print("flatPair\n{0}".format(flatPair))

        # Update labels for pair members
        for i in flatPair:
            labels[i] = j

        # Add new cluster to clusters
        clusters.add(flatPair)

        print("cluster lens\n{0}".format(len(clusters)))

        # End if no more clusters
        if len(clusters) == 1:
            break

        print("LABELS\n{0}".format(labels))
        print("INTERATION\n{0}".format(j))
        # Increment
        j += 1

    print(labels)
    return labels
