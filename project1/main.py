import numpy as np

def main():
    ### input
    x = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = range(12))
    print x
    y = np.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])
    print y

    ### compute optimal weight vector w

    xTx = np.matmul(np.transpose(x), x)
    xTx_inverse = np.linalg.inv(xTx)
    xTy = np.matmul(np.transpose(x), y)

    w = np.matmul(xTx_inverse, xTy)
    print "W: "
    print w


if __name__ == '__main__':
    main()
