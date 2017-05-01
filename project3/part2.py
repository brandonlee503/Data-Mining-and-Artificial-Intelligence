import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

CLASS_VALUES = [-1, 1]

def getGiniIndex(sections, Y):
    """
    Calculate Gini Index to evaluate split cost

    @param sections: All the sections of a divide
    @return: The Gini index for cost of split
    """
    gini = 0.0
    for value in CLASS_VALUES:
        for section in sections:
            sectionSize = len(section)
            if sectionSize == 0:
                continue
            ratio = np.count_nonzero(Y == value) / float(sectionSize)
            # print("np count: " + str(np.count_nonzero(Y == value)))
            # print("sectionssize:" + str(sectionSize))
            # print("ratio: " + str(ratio))
            gini += (ratio * (1.0 - ratio))
    # print("gini: " + str(gini))
    return gini

def getSplit(index, value, data):
    """
    Split data based off of attribute and value

    @param index: Attribute index
    @param value: Attribute value
    @param data: Dataset
    @return: Two np arrays representing the split data
    """
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row[index])
        else:
            right.append(row[index])
    return left, right

def getBestSplit(data, Y):
    """
    Greedily calls getSplit() on every value on every feature, evaluates the cost,
    produce best split.

    @param data: The dataset which is being utilized
    @param Y: Class Value Results
    @return: Dictionary representing best value
    """
    bestIndex, bestValue, bestScore, bestGroups = 100, 100, 100, None
    for i in range(len(data[0])):
        for row in data:
            groups = getSplit(i, row[i], data)
            gini = getGiniIndex(groups, Y)
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = i, row[i], gini, groups
    return {"index": bestIndex, "value": bestValue, "groups": bestGroups}
