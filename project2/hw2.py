# Resources
# http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
# http://machinelearningmastery.com/logistic-regression-for-machine-learning/

import sys
import numpy as np
import csv
from numpy import genfromtxt
from math import exp
from LogisticRegression import LR

LEARNING_RATE = 0.00000001
N_EPOCH = 800

def main():
    lr = LR()
    print 'Loading data...'
    XTrain, YTrain, XTest, YTest = getData()

    print 'Starting problem 2...'
    w = problem2(XTrain, YTrain, XTest, YTest, LEARNING_RATE, N_EPOCH, lr)

    print 'Starting problem 3...'
    problem3(XTrain, YTrain, XTest, YTest, LEARNING_RATE, N_EPOCH, lr)

def problem2(XTrain, YTrain, XTest, YTest, learningRate, nEpoch, lr):
    w = np.zeros(XTrain.shape[1])
    trainCorrectness = []
    testCorrectness = []

    print 'Training model and testing correctness...'
    for epoch in range(nEpoch):
        if epoch in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
            print epoch
        d = lr.computeCoefficient(XTrain, YTrain, w)
        w = w + (learningRate * d)

        trainCorrectness.append(lr.testData(w, XTrain, YTrain))
        testCorrectness.append(lr.testData(w, XTest, YTest))

    nicePrint(trainCorrectness, testCorrectness)
    return w

def problem3(XTrain, YTrain, XTest, YTest, learningRate, nEpoch, lr):
    w = np.zeros(XTrain.shape[1])
    print 'Training model and testing correctness...'
    for regcoefficient in [0.01, 0.1, 1, 10, 100, 1000, 5000]:
        print 'Training with lambda: ' + str(regcoefficient)
        trainCorrectness = []
        testCorrectness = []
        w = np.zeros(XTrain.shape[1])
        for epoch in range(nEpoch):
            if epoch in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
                print epoch
            d = lr.computeCoefficient(XTrain, YTrain, w)
            w = w + (learningRate * (d + regcoefficient))

            trainCorrectness.append(lr.testData(w, XTrain, YTrain))
            testCorrectness.append(lr.testData(w, XTest, YTest))
        nicePrint(trainCorrectness, testCorrectness, 'trainingAccuracyReg' + str(regcoefficient) + '.txt', 'testingAccuracyReg' + str(regcoefficient) + '.txt')

def getData():
    training = genfromtxt('usps-4-9-train.csv', delimiter=',')
    testing = genfromtxt('usps-4-9-test.csv', delimiter=',')

    XTrain = np.array(training[:,:256])
    YTrain = np.array(training[:,256:257])
    XTest = np.array(testing[:,:256])
    YTest = np.array(testing[:,256:257])

    return XTrain, YTrain, XTest, YTest

def nicePrint(trainCorrectness, testCorrectness, trainFile = './trainingAccuracy.txt', testFile = './testingAccuracy.txt'):
    trainCorrectnessFile = open(trainFile, 'w')
    testCorrectnessFile = open(testFile, 'w')
    temp = ''

    for i, result in enumerate(trainCorrectness):
        temp = temp + '(' + str(i) + ',' + str(result) + ')\n'
    trainCorrectnessFile.write(temp)

    temp = ''
    for i, result in enumerate(testCorrectness):
        temp = temp + '(' + str(i) + ',' + str(result) + ')\n'
    testCorrectnessFile.write(temp)

if __name__ == '__main__':
    main()
