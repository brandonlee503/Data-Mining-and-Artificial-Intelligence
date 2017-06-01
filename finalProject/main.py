import sys
import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
import pandas as pd

def main():
    # Seperate data and class values
    XTrain = getData()


# Parse CSV
def getData():
    training = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
    training = training.values
    print(training[0])
    # print(training[0])
    # testing = genfromtxt('test.csv', delimiter=',')
    # XTrain = np.array(training[:,0:5])
    # print(XTrain)
    # YTrain = np.array(training[:,5:6])
    # XTest = np.array(testing[:,1:31])
    # YTest = np.array(testing[:,0:1])
    return training


if __name__ == '__main__':
    main()
