# distances - Distance between all points in the data set
# e.g. distance between row data[149] and data[0] is distances[0][149 - (0 + 1)]
# e.g. distance between row data[12] and data[37] is distances[12][37 - (12 + 1)]

# for i, cluster1 in enumerate(clusters):
#     dist[cluster1] = {}
#     for j in range(i + 1, len(clusters)):
#         dist[cluster1][clusters[j]] = np.linalg.norm(data[i] - data[j])


# NOTE: Single link/comple link might help speed this up immensely. Might be doing
# more work than necessary

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

    def __hash__(self):
        return hash(self.label)

    def calcCenter(self, data):
        self.center = np.zeros(data.shape[1]).flatten()

        for i, dataPoint in enumerate(self.points):
            self.center = np.add(self.center, data[dataPoint])

        self.center = np.divide(self.center, len(self.points))


def part2Test(data):
    print('Starting part 2')
    dist = {}
    clusters = {}
    labelCount = 0
    cluster1 = None
    cluster2 = None

    for point in range(len(data)):
        cluster = Cluster([point], labelCount, data)
        labelCount += 1
        dist[cluster] = {}
        for prevCluster in clusters:
            dist[cluster][prevCluster] = np.linalg.norm(cluster.center - prevCluster.center)
        clusters[cluster] = cluster

    while len(clusters) > 1:
        labelCount += 1
        # Find 2 closest clusters

        cluster1, cluster2 = findClosest(clusters, dist, cluster1, cluster2)
        bestDist = dist[cluster1][cluster2]
        del dist[cluster1]
        del dist[cluster2]

        # Merge 2 closest
        newCluster = merge(cluster1, cluster2, clusters, data, labelCount)
        del clusters[cluster1]
        del clusters[cluster2]

        # Calc new
        dist[newCluster] = {}
        for cluster in clusters:
            dist[newCluster][cluster] = np.linalg.norm(newCluster.center - cluster.center)

        clusters[newCluster] = newCluster

        if len(clusters) < 12:
            print('{0} merged with {1} to form {2}'.format(newCluster.child1.label, newCluster.child2.label, newCluster.label))
            print('Height:\t{0}'.format(len(clusters)))
            print('Dist:\t{0}'.format(bestDist))
            print('{0} size:\t{1}'.format(newCluster.child1.label, len(newCluster.child1.points)))
            print('{0} size:\t{1}'.format(newCluster.child2.label, len(newCluster.child2.points)))
            print('')


def findClosest(clusters, dist, prevC1, prevC2):
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
    newCluster = Cluster(
        cluster1.points + cluster2.points,
        labelCount,
        data,
        child1=copy.deepcopy(cluster1),
        child2=copy.deepcopy(cluster2)
    )

    return newCluster
