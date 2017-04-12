import numpy

def main():
    x = numpy.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = range(12))
    print x
    y = numpy.loadtxt(open("housing_train.txt", "rb"), delimiter=" ", usecols = [13])
    print y

if __name__ == '__main__':
    main()
