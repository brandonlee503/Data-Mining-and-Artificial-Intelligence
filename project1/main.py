import numpy as np
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

def main():

    ### Part 1 - Input

    # Input
    # Training data
    inputArr = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ",
                          usecols = range(13))

    # Number of columns based on rows in input array
    (inputArr_col, _) = inputArr.shape
    ones = np.ones((inputArr_col, 1))

    x_train = np.concatenate((ones, inputArr), axis=1)
    y_train = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])


    # Testing data
    inputArr2 = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ",
                           usecols = range(13))

    (inputArr_col2, _) = inputArr2.shape
    ones_2 = np.ones((inputArr_col2, 1))

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
    print compute_SSE(x_train, y_train, w_train)
    print "Test SSE: "
    print compute_SSE(x_test, y_test, w_test)


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
    print compute_SSE(inputArr, y_train, w_train_wo_dummy)
    print "Test SSE: "
    print compute_SSE(inputArr2, y_test, w_test_wo_dummy)

    ### part 5 -- adding random features
    for i in range (0,8,2):
        # add rand feature columns to data
        for j in range(0,i,2):
            rand_feat_train_col = generate_rand(433)
            x_train = np.concatenate((x_train, rand_feat_train_col), axis=1)
            rand_feat_test_col = generate_rand(74)
            x_test = np.concatenate((x_test, rand_feat_test_col), axis=1)
        print
        print " ---- With "+ str(i) + " random features ----"
        print
        ### compute optimal weight vector w
        print "Training W: "
        w_train = compute_w(x_train, y_train)
        print w_train
        print "Testing W: "
        w_test = compute_w(x_test, y_test)
        print w_test

        ### compute SSE
        print "Training SSE: "
        print compute_SSE(x_train, y_train, w_train)
        print "Test SSE: "
        print compute_SSE(x_test, y_test, w_test)

    ### part 6 --- computing w with lambda
    print
    print "--- PART 6 ---"
    print
    lambda_vals = [0.01, 0.05, 0.1, 0.5, 1, 5]
    
    for lval in lambda_vals:
        print "Training W with lambda: " + str(lval)
        wl_train = computer_w_with_lambda(x_train, y_train, lval)
        wl_test = computer_w_with_lambda(x_test, y_test, lambda_vals[0])
        print "Train SSE: "
        print compute_SSE(x_train, y_train, wl_train)
        print "Test SSE: "
        print compute_SSE(x_test, y_test, wl_test)


def generate_rand(arrLen):
    ### Generate vector of 0's and random values
    dummy = [0] * arrLen
    a = random.randint(0, 10000)
    for i, value in enumerate(dummy):
        coinflip = random.randint(0, 1)
        if coinflip:
            dummy[i] = random.randint(0, a)

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

def computer_w_with_lambda(x, y, l):
    ### compute optimal weight vector w with lambda
    xTx = np.matmul(np.transpose(x), x)
    #create identity
    (xlen, _) = xTx.shape
    iden = np.eye(xlen)
    #mult lamda identity
    iden = np.multiply(l, iden)
    #add identity
    xId = np.add(iden, xTx)
    xId_inverse = np.linalg.inv(xId)
    xTy = np.matmul(np.transpose(x), y)
    w = np.matmul(xId_inverse, xTy)
    return w

def compute_SSE(x, y, w):
    ### compute SSE
    e1 = np.transpose(y - np.matmul(x, w))
    e2 = y - np.matmul(x, w)
    e3 = np.matmul(e1, e2)
    return e3

if __name__ == '__main__':
    main()
