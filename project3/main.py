# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

from part1 import part1Problem2
from part2 import createTree, print_tree

def main():
    # Seperate data and class values
    XTrain, YTrain, XTest, YTest = getData()

    # Normalize Data
    XTrain /=  XTrain.sum(axis=1)[:,np.newaxis]
    XTest /=  XTest.sum(axis=1)[:,np.newaxis]

    # Normalized data with class values
    TrainingData = np.append(XTrain, YTrain, axis=1)
    TestingData = np.append(XTest, YTest, axis=1)

    # part1Problem2(XTrain, YTrain, XTest, YTest)
    # foo = getBestSplit(TrainingData)
    # print(getBestSplit(TrainingData))
    # print('Split: [X%d < %.3f]' % ((foo['index']+1), foo['value']))
    tree = createTree(TrainingData, 1, 1)
    print_tree(tree)

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
