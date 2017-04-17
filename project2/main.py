import numpy as np
import pandas as pd


# TODO: input,work, output
# TODO: ^ make a more descriptive todo

def main():
    (testData_x, testData_y, trainData_x, trainData_y) = importing_data()







# TODO: Test if this really parses data correctly
# TODO: (Possibly) creates a view of original, when view is modified so is
# original
def importing_data():
    print "importing"
    testLoc = '../project2/usps-4-9-test.csv'
    trainLoc = '../project2/usps-4-9-train.csv'

    dfTest = pd.read_csv(testLoc, header=None)
    dfTrain = pd.read_csv(trainLoc, header=None)
    
    dfTestArr = dfTest.values
    dfTrainArr = dfTrain.values

    (_, testSize_col) = dfTestArr.shape
    (_, trainSize_col) = dfTrainArr.shape

    dfTest_x = dfTestArr[:,:-1]
    dfTest_y = dfTestArr[:, testSize_col-1]

    dfTrain_x = dfTrainArr[:,:-1]
    dfTrain_y = dfTrainArr[:, trainSize_col-1]

    return (dfTest_x, dfTest_y, dfTrain_x, dfTrain_y)



def compute_d(x,y,w):
    y_hat = (1 / (1 + exp( -w * x )))
    error = y - y_hat
    d = d + error * x
    return d

def compute_w(x,y):
    ### compute optimal weight vector w
    xTx = np.matmul(np.transpose(x), x)
    xTx_inverse = np.linalg.inv(xTx)
    xTy = np.matmul(np.transpose(x), y)

    w = np.matmul(xTx_inverse, xTy)
    return w

if __name__ == '__main__':
    main()
