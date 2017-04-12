import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

def main():

    ### Part 1 - Input

    # Ones column buffer
    ones = np.ones((433, 1))

    # Input
    inputArr = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = range(12))
    x_train = np.concatenate((ones, inputArr), axis=1)
    y_train = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])
    # x_test = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = range(12))
    # y_test = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = [13])


    ### Part 2 - compute optimal weight vector w
    print "Training W: "
    w = compute_w(x_train, y_train)
    print w

    # print "Testing W: "
    # print compute_w(x_test, y_test)
    print compSSE(x_train, y_train, w)



def compute_w(x,y):
    ### compute optimal weight vector w
    xTx = np.matmul(np.transpose(x), x)
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
