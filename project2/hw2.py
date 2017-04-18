import sys
import numpy as np
import csv
from numpy import genfromtxt

def main():
    XTrain, YTrain, XTest, YTest = getData()

def getData():
    trainData = genfromtxt('usps-4-9-train.csv', delimiter=',')
    testData = genfromtxt('usps-4-9-train.csv', delimiter=',')

    XTrain = np.array(trainData[:,:256])
    YTrain = np.array(trainData[:,256:257])
    XTest = np.array(testData[:,:256])
    YTest = np.array(testData[:,256:257])

    return XTrain, YTrain, XTest, YTest

if __name__ == '__main__':
    main()
