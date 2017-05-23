import numpy as np
import math

from shared import dist

MAX_ITER = 1000

def part1(data):
    print('\nStarting Problem 1.1')
    SSEs, labels, centers, iterations = kmeans(data, k=2)
    totalSSEs = []
    for i in range(iterations):
        tempVal = 0
        for j, SSE in enumerate(SSEs):
            tempVal += SSE[i]
        totalSSEs.append(tempVal)

    for SSE in totalSSEs:
        print(SSE)

    print('\nProblem 1.2 disabled due to time requirements')
    # minSSEs = []
    #
    # for k in range(2, 11):
    #     minSSE = math.inf
    #     for i in range(10):
    #         SSEs, labels, centers, iterations = kmeans(data, k=k)
    #         minSSE = min(calcTotalSSE(SSEs, iterations - 1), minSSE)
    #     minSSEs.append(minSSE)
    #
    # for minSSE in minSSEs:
    #     print(minSSE)


def calcTotalSSE(SSEs, iteration):
    tempVal = 0

    for j, SSE in enumerate(SSEs):
        tempVal += SSE[iteration]
    return tempVal


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
