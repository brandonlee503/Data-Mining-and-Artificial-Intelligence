import numpy as np
from numpy import genfromtxt

def dist(row, center):
    """
    Calculates euclidean distance between two data points

    @param row: The data point
    @param center: The cluster center
    @returns Float representation of euclidean distance
    """
    return np.linalg.norm(row - center)

def getData(fName='data-1.txt'):
    """
    Parses CSV for relevant data
    """
    print('Loading data...')
    return genfromtxt(fName, delimiter=',')
