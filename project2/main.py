import numpy as np




# TODO: input,work, output

def main():
    print "main"







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
