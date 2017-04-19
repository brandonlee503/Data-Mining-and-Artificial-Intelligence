# Resources
# http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
# http://machinelearningmastery.com/logistic-regression-for-machine-learning/
import sys
import numpy as np
import csv
from numpy import genfromtxt
from math import exp
from LogisticRegression import LR

LEARNING_RATE = 0.000000001
N_EPOCH = 900

def main():
    lr = LR()
    print 'Loading data...'
    XTrain, YTrain, XTest, YTest = getData()
    print 'Starting problem 2...'
    problem2(XTrain, YTrain, XTest, YTest, LEARNING_RATE, N_EPOCH, lr)

def problem2(XTrain, YTrain, XTest, YTest, learningRate, nEpoch, lr):
    w = np.zeros(XTrain.shape[1])
    trainCorrectness = []
    testCorrectness = []

    print 'Training model and testing correctness...'
    for epoch in range(nEpoch):
        d = lr.computeCoefficient(XTrain, YTrain, w)
        w = w + (learningRate * d)

        trainCorrectness.append(lr.testData(w, XTrain, YTrain))
        testCorrectness.append(lr.testData(w, XTest, YTest))

def getData():
    training = genfromtxt('usps-4-9-train.csv', delimiter=',')
    testing = genfromtxt('usps-4-9-test.csv', delimiter=',')

    XTrain = np.array(training[:,:256])
    YTrain = np.array(training[:,256:257])
    XTest = np.array(testing[:,:256])
    YTest = np.array(testing[:,256:257])

    return XTrain, YTrain, XTest, YTest

if __name__ == '__main__':
    main()
