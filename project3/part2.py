# References
# http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/

import sys
import numpy as np
import csv
import math
from numpy import genfromtxt

CLASS_VALUES = [-1.0, 1.0]

def getGiniIndex(sections):
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
            ratio = [classRow[-1] for classRow in section].count(value) / float(sectionSize)
            gini += (ratio * (1.0 - ratio))
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
            left.append(row)
        else:
            right.append(row)
    return left, right

def getBestSplit(data):
    """
    Greedily calls getSplit() on every value on every feature, evaluates the cost,
    produce best split.

    @param data: The dataset which is being utilized
    @param Y: Class Value Results
    @return: Dictionary representing best value
    """
    bestIndex, bestValue, bestScore, bestGroups = 100, 100, 100, None
    # All the columns
    for i in range(len(data[0]) - 1):
        # All the rows
        for row in data:
            groups = getSplit(i, row[i], data)
            gini = getGiniIndex(groups)
            print('X%d < %.3f Gini=%.3f' % ((i+1), row[i], gini))
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = i, row[i], gini, groups
    return {"index": bestIndex, "value": bestValue, "groups": bestGroups}

def setMajorityClass(section):
    """
    Select class value for section of rows. This is the majority class.

    @param section: A subsection of rows
    @return: Most common result class
    """
    majority = [row[-1] for row in section]
    return max(key=majority.count, set(majority))
