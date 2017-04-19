# Resources
# http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
# http://machinelearningmastery.com/logistic-regression-for-machine-learning/
import sys
import numpy as np
import csv
from numpy import genfromtxt
from math import exp

LEARNING_RATE = 0.000000001
N_EPOCH = 10

def main():
    XTrain, YTrain, XTest, YTest = getData()
    computeCoefficient(XTrain, YTrain, LEARNING_RATE, N_EPOCH)

def getData():
    trainData = genfromtxt('usps-4-9-train.csv', delimiter=',')
    testData = genfromtxt('usps-4-9-train.csv', delimiter=',')

    XTrain = np.array(trainData[:,:256])
    YTrain = np.array(trainData[:,256:257])
    XTest = np.array(testData[:,:256])
    YTest = np.array(testData[:,256:257])

    return XTrain, YTrain, XTest, YTest

# Logistic regression prediction
def computePrediction(row, w):
    return 1.0 / (1.0 + np.exp(-1.0 * (np.dot(w, row))))

# Calculate coefficients w/ batch stochastic gradient descent
def computeCoefficient(trainingX, trainingY, learningRate, nEpoch):
    w = np.zeros(trainingX.shape[1])
    for epoch in range(nEpoch):
        d = np.zeros(trainingX.shape[1])
        for row in range(len(trainingX)):
            yHat = computePrediction(trainingX[row], w)
            error = float(trainingY[row]) - yHat
            d = np.add(d, np.multiply(error, trainingX[row]))
        w = w + (learningRate*d)
    return w

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
