import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

MAX_ITER = 20

def main():
    data = getData()
    # kmeans(data, k=2)

    #### TESTING
    print(getLabels(data, randCenters(data, 2)))
    ####


def kmeans(data, k=2):
    count = 0
    centers = randCenters(data, k)
    oldCenters = None
    labels = []

    print(centers)
    while not converged(oldCenters, centers, count):
        oldCenters = centers
        count += 1

        labels = getLabels(data, centers)
        centers = getCenters(data, labels, centers)


def randCenters(data, k):
    return [data[np.random.randint(0, len(data), size=1)].flatten().tolist() for i in range(k)]


def converged(oldCenters, centers, count):
    return True if count >= MAX_ITER or oldCenters == centers else False


# Parse CSV
def getData():
    return genfromtxt('data-1.txt', delimiter=',')


def getLabels(data, centers):
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
    return np.linalg.norm(row - center)


def getCenters(data, labels, centers):
    sums = tuple([np.zeroes(len(centers), data.shape[1])], 0)
    newCenters = []

    for i, label in enumerate(labels):
        sums[label][0] += data[i]
        sums[label][1] += 1
    for i, row in enumerate(sums):
        newCenters.append(row[0] / row[1])

    return newCenters


if __name__ == '__main__':
    main()
