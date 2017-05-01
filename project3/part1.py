# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

K = range(1, 52, 2)

def part1Problem2(XTrain, YTrain, XTest, YTest):
    testResults = []
    trainResults = []
    crossValidation = []

    # Compute for training set
    computeCorrect(XTrain, YTrain, XTrain, YTrain, trainResults)
    # Compute for testing set
    computeCorrect(XTrain, YTrain, XTest, YTest, testResults)

    # Compute for leave out one cross validation
    for i, row in enumerate(XTrain):
        crossValidation.append([])
        computeCorrect(XTrain, YTrain, XTest, YTest, crossValidation[i])
        print(i)

# Compares model with dataset and calculates number of correct rows
def computeCorrect(XTrain, YTrain, x, y, results):
    knn = [getKNN(XTrain, YTrain, row) for row in x]

    for i, k in enumerate(K):
        correct = 0

        for j, row in enumerate(knn):
            result = sum(pair[0] for pair in row[:k])
            if result > 0 and y[j] == 1 or result < 0 and y[j] == -1:
                correct += 1
        results.append(correct)

    return results


# Returns k neighbors of given set
def getKNN(XTrain, YTrain, testRow):
    distances = [(*YTrain[i], getDistance(testRow, trainRow)) for i, trainRow in enumerate(XTrain)]
    distances.sort(key=lambda tup: tup[1])

    return distances


# Gets the distance between two points
def getDistance(x, xi):
    sumDistance = 0
    for j in range(len(x)):
        sumDistance += pow((x[j] - xi[j]), 2)

    return math.sqrt(sumDistance)


# Parse CSV
def getData():
    training = genfromtxt('knn_train.csv', delimiter=',')
    testing = genfromtxt('knn_test.csv', delimiter=',')
    XTrain = np.array(training[:,1:31])
    YTrain = np.array(training[:,0:1])
    XTest = np.array(testing[:,1:31])
    YTest = np.array(testing[:,0:1])
    return XTrain, YTrain, XTest, YTest
