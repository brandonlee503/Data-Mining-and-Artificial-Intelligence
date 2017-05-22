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
    Calculate Gini Index to evaluate split cost. This is the information gain.

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
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = i, row[i], gini, groups

    return {"index": bestIndex, "value": bestValue, "groups": bestGroups, "infoGain": bestScore}

def setMajorityClass(section):
    """
    Select class value for section of rows. This is the majority class.

    @param section: A subsection of rows
    @return: Most common result class
    """
    majority = [row[-1] for row in section]
    return max(set(majority), key=majority.count)

def splitTree(node, currentDepth, maxDepth, minSize):
    """
    Recursively builds the decision tree through best splitting and
    based on a variety of parameters.

    @param node: Initial stump
    @param currentDepth: The current level
    @param maxDepth: Maximum depth before setting majority class
    @param minSize: Minimum # of rows before setting majority class
    @returns: A recrusively built tree
    """
    left, right = node["groups"]
    del(node["groups"])
    # Check if there's no split
    if not left or not right:
        node["left"] = node["right"] = setMajorityClass(left + right)
        return
    # Check if max depth has been reached
    if maxDepth <= currentDepth:
        node["left"], node["right"] = setMajorityClass(left), setMajorityClass(right)
        return
    # Check left child
    if len(left) <= minSize:
        node["left"] = setMajorityClass(left)
    else:
        node["left"] = getBestSplit(left)
        splitTree(node["left"], currentDepth + 1, maxDepth, minSize)
    # Check right child
    if len(right) <= minSize:
        node["right"] = setMajorityClass(right)
    else:
        node["right"] = getBestSplit(right)
        splitTree(node["right"], currentDepth + 1, maxDepth, minSize)


def createTree(data, maxDepth, minSize):
    """
    Initializes tree with root node and recrusively calls splitTree() to build
    decision tree for specified criteria.

    @param data: The dataset
    @param maxDepth: Maximum depth before setting majority class
    @param minSize: Minimum # of rows before setting majority class
    @returns: A decision tree
    """
    root = getBestSplit(data)
    splitTree(root, 1, maxDepth, minSize)
    return root


def getErrorRate(tree, data):
    """
    Utilizes decision tree to check decision tree accuracy.

    @param tree: The decision tree
    @param data: The dataset
    @returns: The error percentage
    """
    correct = 0
    for row in data:
        prediction = makePrediction(tree, row)
        if prediction == row[-1]:
            correct += 1
    return 1 - (float(correct) / len(data))


def makePrediction(treeNode, row):
    """
    Recursively iterate through three to make a prediction.

    @param treeNode: The current node in decision tree
    @param row: The current row in data set
    @return: The prediction
    """
    if row[treeNode["index"]] < treeNode["value"]:
        if isinstance(treeNode["left"], dict):
            return makePrediction(treeNode["left"], row)
        else:
            return treeNode["left"]
    else:
        if isinstance(treeNode["right"], dict):
            return makePrediction(treeNode["right"], row)
        else:
            return treeNode["right"]

def printTree(treeNode, depth=0):
    """
    Prints decision tree for visual observation
    http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

    @param treeNode: The decision tree node
    @param depth: Current depth
    """
    if isinstance(treeNode, dict):
        print "%s{X%d = %.4f}" % ((depth*str(depth), (treeNode["index"] + 1), treeNode["value"]))
        printTree(treeNode["left"], depth + 1)
        printTree(treeNode["right"], depth + 1)
    else:
        print "%s{%s}" % ((depth * "*", treeNode))

def printInfoGain(treeNode, depth=0):
    """
    Iterates through the decision tree and prints all the gini values.

    @param treeNode: The decision tree node
    @param depth: Current depth
    """
    if isinstance(treeNode, dict):
        print "Depth: " + str(depth) + " | Information Gain: " + str(treeNode["infoGain"])
        printInfoGain(treeNode["left"], depth + 1)
        printInfoGain(treeNode["right"], depth + 1)

def createDecisionStump(trainingData, testingData):
    """
    Part 2 Problem 1 - Create decision stump and display associated values

    @param trainingData: The training dataset
    @param testingData: The testing dataset
    """
    tree = createTree(trainingData, 1, 10)
    print "PART 2 PROBLEM 1"
    print "Learned Stump:"
    printTree(tree)
    print "\n"
    print "Information Gain: " + str(tree["infoGain"])
    print "Training Error Rate: " + str(getErrorRate(tree, trainingData))
    print "Testing Error Rate: " + str(getErrorRate(tree, testingData))
    print "\n"

def createDecisionTree(trainingData, testingData):
    """
    Part 2 Problem 2 - Create decision tree and display associated values

    @param trainingData: The training dataset
    @param testingData: The testing dataset
    """
    # Max depth of 6
    tree = createTree(trainingData, 6, 10)
    print "PART 2 PROBLEM 2"
    print "Learned Tree:"
    printTree(tree)
    print "\n"
    # print("Information Gain: " + str(tree["infoGain"]))
    printInfoGain(tree)
    print "Training Error Rate: " + str(getErrorRate(tree, trainingData))
    print "Testing Error Rate: " + str(getErrorRate(tree, testingData))
    print "\n"
