import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

def main():
    XTrain, YTrain, XTest, YTest = getData()
    print getKNN(3, XTrain, XTest[0])


def getKNN(k, XTrain, testRow):
    distances = [(trainRow, distance(testRow, trainRow)) for i, trainRow in enumerate(XTrain)]
    distances.sort(key=lambda tup: tup[1])
    return [distances[i][0] for i in range(k)]

def getData():
    training = genfromtxt('knn_train.csv', delimiter=',')
    testing = genfromtxt('knn_test.csv', delimiter=',')

    XTrain = np.array(training[:,1:31])
    YTrain = np.array(training[:,0:1])
    XTest = np.array(testing[:,1:31])
    YTest = np.array(testing[:,0:1])

    return XTrain, YTrain, XTest, YTest


def distance(x, xi):
    sumDistance = 0
    for j in range(len(x)):
        sumDistance += pow((x[j] - xi[j]), 2)
    return math.sqrt(sumDistance)

if __name__ == '__main__':
    main()
