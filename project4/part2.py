import numpy as np
import copy

class Cluster(object):
    """docstring for Cluster."""
    def __init__(self, points, label, data, child1=None, child2=None, parent=None):
        self.points = points
        self.label = label
        self.center = np.zeros(data.shape[1]).flatten()
        self.calcCenter(data)
        self.child1 = child1
        self.child2 = child2
        self.parent = parent

    def __eq__(self, other):
        return hasattr(other, 'label') and self.label == other.label

    def __gt__(self, other):
        return other.label < self.label

    def __lt__(self, other):
        return other.label > self.label

    def __hash__(self):
        return hash(self.label)

    def calcCenter(self, data):
        self.center = np.zeros(data.shape[1]).flatten()

        for i, dataPoint in enumerate(self.points):
            self.center = np.add(self.center, data[dataPoint])

        self.center = np.divide(self.center, len(self.points))


def part2(data):
    """
    Runs hacmin and hacmax

    @param data: List of data points
    """

    print('\n\nStarting part 2')
    dist = {}
    clusters = {}
    labelCount = 0

    print('Finding distance between points')
    for point in range(len(data)):
        cluster = Cluster([point], labelCount, data)
        labelCount += 1
        dist[cluster] = {}
        for prevCluster in clusters:
            dist[cluster][prevCluster] = np.linalg.norm(cluster.center - prevCluster.center)
        clusters[cluster] = cluster

    clusters2 = copy.deepcopy(clusters)
    dist2 = copy.deepcopy(dist)

    print('Starting 2.1')
    hacMin(clusters, dist, data, labelCount)
    print('#################### Done ####################')

    print('\nStarting 2.2')
    hacMax(clusters2, dist2, data, labelCount)


def hacMin(clusters, dist, data, labelCount):
    """
    Performs hac with single-link clustering

    @param clusters: dictionary containing all current clusters
    @param dist: dictionary containing lookups for distances between all clusters
    @param data: List of data points
    @param labelCount: Counter for labeling clusters
    @returns: Classified cluster data points
    """

    cluster1 = None
    cluster2 = None

    while len(clusters) > 1:
        labelCount += 1

        # Find 2 closest clusters
        cluster1, cluster2 = findClosest(clusters, dist, cluster1, cluster2)
        bestDist = dist[cluster1][cluster2]
        # Merge 2 closest
        newCluster = merge(cluster1, cluster2, clusters, data, labelCount)

        minDist(clusters, newCluster, cluster1, cluster2, dist)

        del dist[cluster1]
        del dist[cluster2]
        del clusters[cluster1]
        del clusters[cluster2]

        clusters[newCluster] = newCluster

        if len(clusters) < 12:
            printInfo(clusters, newCluster, bestDist)


def hacMax(clusters, dist, data, labelCount):
    """
    Performs hac with complete-link clustering

    @param clusters: dictionary containing all current clusters
    @param dist: dictionary containing lookups for distances between all clusters
    @param data: List of data points
    @param labelCount: Counter for labeling clusters
    @returns: Classified cluster data points
    """
    cluster1 = None
    cluster2 = None

    while len(clusters) > 1:
        labelCount += 1

        # Find 2 closest clusters
        cluster1, cluster2 = findClosest(clusters, dist, cluster1, cluster2)
        bestDist = dist[cluster1][cluster2]
        # Merge 2 closest
        newCluster = merge(cluster1, cluster2, clusters, data, labelCount)

        maxDist(clusters, newCluster, cluster1, cluster2, dist)

        del dist[cluster1]
        del dist[cluster2]
        del clusters[cluster1]
        del clusters[cluster2]

        clusters[newCluster] = newCluster

        if len(clusters) < 12:
            printInfo(clusters, newCluster, bestDist)


def printInfo(clusters, newCluster, bestDist):
    """
    Prints info about what clusters are merging

    @param clusters: dictionary containing all current clusters
    @param newCluster: Cluster being merged
    @param bestDist: Distance of the two clusters being merged
    """

    # print('{0} merged with {1} to form {2}'.format(newCluster.child1.label, newCluster.child2.label, newCluster.label))
    print('{0} & {1} & {2} & {3} & {4} & {5} & {6} \\\\ \\hline'.format(
        newCluster.child1.label,
        newCluster.child2.label,
        newCluster.label,
        len(clusters),
        bestDist,
        len(newCluster.child1.points),
        len(newCluster.child2.points)
    ))
    # print('Height: {0}, Dist: {1}, Size of {2}: {3}, Size of {4}: {5}'.format(len(clusters), bestDist, newCluster.child1.label, len(newCluster.child1.points), newCluster.child2.label, len(newCluster.child2.points)))
    # print('Dist:\t{0}'.format(bestDist))
    # print('{0} size:\t{1}'.format(newCluster.child1.label, len(newCluster.child1.points)))
    # print('{0} size:\t{1}'.format(newCluster.child2.label, len(newCluster.child2.points)))
    # print('')


def findClosest(clusters, dist, prevC1, prevC2):
    """
    Finds the two closest clusters

    @param clusters: dictionary containing all current clusters
    @param dist: dictionary containing lookups for distances between all clusters
    @param prevC1: One of previous clusters merged. Used for deleting entry in dist
    @param prevC2: One of previous clusters merged. Used for deleting entry in dist
    @returns: 2 closest clusters
    """

    cluster1 = None
    cluster2 = None
    best = np.inf

    for i, c1 in enumerate(clusters):
        if prevC1 in dist[c1]:
            del dist[c1][prevC1]
        if prevC2 in dist[c1]:
            del dist[c1][prevC2]

        for j, c2 in enumerate(dist[c1]):
            if dist[c1][c2] < best:
                best = dist[c1][c2]
                cluster1 = c1
                cluster2 = c2

    return cluster1, cluster2


def merge(cluster1, cluster2, clusters, data, labelCount):
    """
    Creates new cluster from two other clusters

    @param Cluster1: One of clusters being merged
    @param Cluster2: One of clusters being merged
    @param clusters: dictionary containing all current clusters
    @param data: List of data points
    @param labelCount: Counter for labeling clusters
    @returns: New cluster made from 2 other clusters
    """

    newCluster = Cluster(
        cluster1.points + cluster2.points,
        labelCount,
        data,
        child1=copy.deepcopy(cluster1),
        child2=copy.deepcopy(cluster2)
    )

    return newCluster


def minmax(x, y):
    """
    Finds the min and max of two things

    @param x: thing1
    @param y: thing2
    @returns: min and max
    """

    return min(x, y), max(x, y)


def minDist(clusters, newCluster, cluster1, cluster2, dist):
    """
    Finds minimum distance between clusters being merged and all other clusters

    @param clusters: dictionary containing all current clusters
    @param newCluster: Cluster being made from cluster1 and cluster2
    @param cluster1: One of clusters being merged
    @param cluster2: One of clusters being merged
    @param dist: dictionary containing lookups for distances between all clusters
    """
    dist[newCluster] = {}
    for cluster in clusters:
        if cluster.label == cluster1.label or cluster.label == cluster2.label: continue
        c1Min, c1Max = minmax(cluster, cluster1)
        c2Min, c2Max = minmax(cluster, cluster2)

        dist[newCluster][cluster] = min(dist[c1Max][c1Min], dist[c2Max][c2Min])


def maxDist(clusters, newCluster, cluster1, cluster2, dist):
    """
    Finds maximum distance between clusters being merged and all other clusters

    @param clusters: dictionary containing all current clusters
    @param newCluster: Cluster being made from cluster1 and cluster2
    @param cluster1: One of clusters being merged
    @param cluster2: One of clusters being merged
    @param dist: dictionary containing lookups for distances between all clusters
    """
    dist[newCluster] = {}
    for cluster in clusters:
        if cluster.label == cluster1.label or cluster.label == cluster2.label: continue
        c1Min, c1Max = minmax(cluster, cluster1)
        c2Min, c2Max = minmax(cluster, cluster2)

        dist[newCluster][cluster] = max(dist[c1Max][c1Min], dist[c2Max][c2Min])


def hacAverage(clusters, newCluster, dist):
    """
    Finds distance between two given clusters centers

    @param clusters: dictionary containing all current clusters
    @param newCluster: Cluster being made from cluster1 and cluster2
    @param dist: dictionary containing lookups for distances between all clusters
    """
    for cluster in clusters:
        dist[newCluster][cluster] = np.linalg.norm(newCluster.center - cluster.center)
