# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

K = range(1, 52, 2)

def part1Problem2(train, test):
    testResults = []
    trainResults = []
    crossValidation = []

    # Compute for training set
    computeCorrect(train, train, trainResults)
    # Compute for testing set
    computeCorrect(train, test, testResults)

    nicePrint(trainResults, './trainingAccuracy.txt')
    nicePrint(testResults, './testingAccuracy.txt')

    # Compute for leave out one cross validation
    for i, row in enumerate(train):
        temp = np.copy(row)
        crossValidation.append([])

        train = np.delete(train, i, 0)
        computeCorrectLeaveOneOut(train, temp, crossValidation[i])
        train = np.insert(train, i, temp, 0)

    percentData = [val / 283 for val in np.sum(crossValidation, axis=0)]
    nicePrint(percentData, './leaveoneoutAccuracy.txt')


def computeCorrectLeaveOneOut(train, test, results):
    knn = getKNN(train, test)

    for i, k in enumerate(K):
        correct = 0
        result = sum(pair[0] for pair in knn[:k])
        if result > 0 and test[-1] == 1 or result < 0 and test[-1] == -1:
            correct += 1
        results.append(correct)

    return results


# Compares model with dataset and calculates number of correct rows
def computeCorrect(train, test, results):
    knn = [getKNN(train, row) for row in test]

    for i, k in enumerate(K):
        correct = 0

        for j, row in enumerate(knn):
            result = sum(pair[0] for pair in row[:k])
            if result > 0 and test[j][-1] == 1 or result < 0 and test[j][-1] == -1:
                correct += 1
        print(correct)
        results.append(correct / len(train))

    return results


# Returns k neighbors of given set
def getKNN(train, test):
    distances = []
    for i, trainRow in enumerate(train[:,0:-1]):
        distances.append((train[i][-1], getDistance(test[0:-1], trainRow)))
    # distances = [(*train[i][-1], getDistance(test[0:-1], trainRow)) for i, trainRow in enumerate(train[:,0:-1])]
    distances.sort(key=lambda tup: tup[1])

    return distances


# Gets the distance between two points
def getDistance(x, xi):
    sumDistance = 0
    for j, feature in enumerate(x):
        sumDistance += pow((feature - xi[j]), 2)

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

def nicePrint(data, output='./error.txt'):
    fp = open(output, 'w')
    temp = ''

    for i, result in enumerate(data):
        temp = temp + '(' + str(K[i]) + ',' + str(result) + ')\n'
    fp.write(temp)
