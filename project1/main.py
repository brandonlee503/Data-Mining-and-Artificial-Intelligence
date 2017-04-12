import numpy as np
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

def main():

    ### Part 1 - Input

    # Ones column buffer
    ones = np.ones((433, 1))
    ones_2 = np.ones((74, 1))

    # Input
    inputArr = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = range(12))
    x_train = np.concatenate((ones, inputArr), axis=1)
    y_train = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])

    inputArr2 = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = range(12))
    x_test = np.concatenate((ones_2, inputArr2), axis=1)
    y_test = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = [13])

    print
    print " ---- With dummy ones ----"
    print
    ### Part 2 - compute optimal weight vector w
    print "Training W: "
    w_train = compute_w(x_train, y_train)
    print w_train
    print "Testing W: "
    w_test = compute_w(x_test, y_test)
    print w_test

    ### part 3 -- compute SSE
    print "Training SSE: "
    print compSSE(x_train, y_train, w_train)
    print "Test SSE: "
    print compSSE(x_test, y_test, w_test)


    ### part 4 -- W and SSE without dummy 1s
    print
    print " ---- Without dummy ones ----"
    print
    ### compute optimal weight vector w
    print "Training W: "
    w_train_wo_dummy = compute_w(inputArr, y_train)
    print w_train_wo_dummy
    print "Testing W: "
    w_test_wo_dummy = compute_w(inputArr2, y_test)
    print w_test_wo_dummy

    ### compute SSE
    print "Training SSE: "
    print compSSE(inputArr, y_train, w_train_wo_dummy)
    print "Test SSE: "
    print compSSE(inputArr2, y_test, w_test_wo_dummy)

    ### part 5 -- adding random features
    for i in range (0,4,2):
        # add rand feature columns to data
        for j in range(0,i,2):
            rand_feat_train_col = np.full((433, 1), j)
            x_train = np.concatenate((x_train, rand_feat_train_col), axis=1)
            rand_feat_test_col = np.full((74, 1), j)
            x_test = np.concatenate((x_test, rand_feat_test_col), axis=1)
        print
        print " ---- With "+ str(i) + " random features ----"
        print
        ### Part 2 - compute optimal weight vector w
        print "Training W: "
        w_train = compute_w(x_train, y_train)
        print w_train
        print "Testing W: "
        w_test = compute_w(x_test, y_test)
        print w_test

        ### part 3 -- compute SSE
        print "Training SSE: "
        print compSSE(x_train, y_train, w_train)
        print "Test SSE: "
        print compSSE(x_test, y_test, w_test)


def generateDummy(arrLen):
    dummy = [0] * arrLen
    for i, value in enumerate(dummy):
        coinflip = random.randint(0, 1)
        if coinflip:
            dummy[i] = random.randint(0, 500)

    foo = np.asarray(dummy)[np.newaxis]
    return foo.T


def compute_w(x,y):
    ### compute optimal weight vector w
    xTx = np.matmul(np.transpose(x), x)
    print ("shape: ")
    print xTx.shape
    xTx_inverse = np.linalg.inv(xTx)
    xTy = np.matmul(np.transpose(x), y)

    w = np.matmul(xTx_inverse, xTy)
    return w

def compSSE(x, y, w):
    ### compute SSE
    e1 = np.transpose(y - np.matmul(x, w))
    e2 = y - np.matmul(x, w)
    e3 = np.matmul(e1, e2)
    return e3

if __name__ == '__main__':
    main()
