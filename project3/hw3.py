# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

K = range(1, 52, 2)
CLASS_VALUES = [-1, 1]

def main():
    XTrain, YTrain, XTest, YTest = getData()
    # part1Problem2(XTrain, YTrain, XTest, YTest)
    # print(XTrain)
    print(getSplit(0, 15, XTrain))

### PART 1

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
        print(crossValidation[i])


    print(testResults)

# Compares model with dataset and calculates number of correct rows
def computeCorrect(XTrain, YTrain, x, y, results):
    for i, k in enumerate(K):
        knn = [getKNN(k, XTrain, YTrain, row) for row in x]
        correct = 0

        for j, row in enumerate(knn):
            result = sum(row)
            if result > 0 and y[j] == 1 or result < 0 and y[j] == -1:
                correct += 1
        results.append(correct)

    return results

# Returns k neighbors of given set
def getKNN(k, XTrain, YTrain, testRow):
    distances = [(*YTrain[i], getDistance(testRow, trainRow)) for i, trainRow in enumerate(XTrain)]
    distances.sort(key=lambda tup: tup[1])
    return [distances[i][0] for i in range(k)]

# Gets the distance between two points
def getDistance(x, xi):
    sumDistance = 0
    for j in range(len(x)):
        sumDistance += pow((x[j] - xi[j]), 2)

    return math.sqrt(sumDistance)

### PART 2

def getGiniIndex(sections):
    """
    Calculate GiniIndex to evaluate split cost

    @param sections: All the sections of a divide
    @return: The Gini index for cost of split
    """
    gini = 0.0
    for value in CLASS_VALUES:
        for section in sections:
            sectionSize = len(section)
            if sectionSize == 0:
                continue
            ratio = YTrain.count(value) / float(sectionSize)
            gini += (ratio * (1.0 - ratio))
    return gini

def getSplit(index, value, data):
    """
    Split data based off of attribute and value

    @param index: Attribute index
    @param value: Attribute value
    @param data: Dataset
    @return: Two np arrays representing the split data
    """
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row[index])
        else:
            right.append(row[index])
    return left, right

def getBestSplit(data):
    """
    Greedily calls getSplit() on every value on every feature, evaluates the cost,
    produce best split.

    @param data: The dataset which is being utilized
    @return: Dictionary representing best value
    """
    bestIndex, bestValue, bestScore, bestGroups = 100, 100, 100, None
    for i in range(len(data[0] - 1)):
        for row in data:
            groups = getSplit(i, row[i], data)
            gini = getGiniIndex(groups)
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = i, row[i], gini, groups
    return {"index": bestIndex, "value": bestValue, "groups": bestGroups}

# Parse CSV
def getData():
    training = genfromtxt('knn_train.csv', delimiter=',')
    testing = genfromtxt('knn_test.csv', delimiter=',')
    XTrain = np.array(training[:,1:31])
    YTrain = np.array(training[:,0:1])
    XTest = np.array(testing[:,1:31])
    YTest = np.array(testing[:,0:1])
    return XTrain, YTrain, XTest, YTest

if __name__ == '__main__':
    main()
