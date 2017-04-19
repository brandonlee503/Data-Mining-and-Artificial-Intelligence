# Resources
# http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/

import sys
import numpy as np
import csv
from numpy import genfromtxt
from math import exp

LEARNING_RATE = 0.3

def main():
    XTrain, YTrain, XTest, YTest = getData()
    computeCoefficient(XTrain, YTrain, LEARNING_RATE, 2)
    # batchLogisticRegression(XTrain, YTrain, LEARNING_RATE)

def getData():
    trainData = genfromtxt('usps-4-9-train.csv', delimiter=',')
    testData = genfromtxt('usps-4-9-train.csv', delimiter=',')

    XTrain = np.array(trainData[:,:256])
    YTrain = np.array(trainData[:,256:257])
    XTest = np.array(testData[:,:256])
    YTest = np.array(testData[:,256:257])

    return XTrain, YTrain, XTest, YTest


# def batchLogisticRegression(x, y, lr):
#     numFeatures = x.shape[1]
#     w = np.zeros(numFeatures)
#     j = 0
#     while(1):
#         d = np.zeros(numFeatures)
#         for i, yi in enumerate(y):
#             wxi = np.dot(w, x[i,:])
#             nwxi = wxi * -1
#             denom = 1 + np.exp(nwxi)
#             yihat = 1/denom
#             print yihat
#             error = yi - yihat
#             d = d + np.multiply(error, x[i,:])
#         w = w + np.multiply(lr, d)
#         j = j + 1
#         # Break after j iterations
#         if(j >= 10):
#             break;
#     return w


# Logistic regression prediction
def computePrediction(row, w):
    dot = np.dot(w, row)
    print dot
    exp = np.exp(-1.0 * dot)
    print exp
    return 1.0 / (1.0 + np.exp(-1.0 * (np.dot(w, row))))


# Calculate coefficients w/ batch stochastic gradient descent
def computeCoefficient(trainingX, trainingY, learningRate, nEpoch):
    w = np.zeros(trainingX.shape[1])
    for epoch in range(nEpoch):
        d = np.zeros(trainingX.shape[1])
        # print("foo: " + str(epoch))
        for row in range(len(trainingX)):
            yHat = computePrediction(trainingX[row], w)
            print yHat
            error = float(trainingY[row]) - yHat
            # print "error: " + str(error)
            # print trainingX[row].shape
            # print np.multiply(error, trainingX[row])
            # d = d + (np.multiply(error, trainingX[row]))
            # print "foo"
            # print d.shape
            d = np.add(d, np.multiply(error, trainingX[row]))
        w = w + (learningRate*d)

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef




if __name__ == '__main__':
    main()
