# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

from part1 import part1Problem2
from part2 import getGiniIndex, getSplit, getBestSplit

K = range(1, 52, 2)
CLASS_VALUES = [-1, 1]

def main():
    XTrain, YTrain, XTest, YTest = getData()

    # part1Problem2(XTrain, YTrain, XTest, YTest)
    # foo = getBestSplit(XTrain, YTrain)
    print(getBestSplit(XTrain, YTrain))
    # print('Split: [X%d < %.3f]' % ((foo['index']+1), foo['value']))

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
