import numpy as np

def main():
    ### input
    x_train = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = range(12))
    y_train = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])
    # x_test = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = range(12))
    # y_test = np.loadtxt(open("housing_test.txt", "rb"), delimiter=" ", usecols = [13])

    print "Training W: "
    print compute_w(x_train, y_train)

    # print "Testing W: "
    # print compute_w(x_test, y_test)



def compute_w(x,y):
    ### compute optimal weight vector w
    xTx = np.matmul(np.transpose(x), x)
    xTx_inverse = np.linalg.inv(xTx)
    xTy = np.matmul(np.transpose(x), y)

    w = np.matmul(xTx_inverse, xTy)
    return w

if __name__ == '__main__':
    main()
