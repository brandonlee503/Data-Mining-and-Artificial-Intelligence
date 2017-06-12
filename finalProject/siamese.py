# Brandon Lee / John Miller
# References:
# http://www.erogol.com/duplicate-question-detection-deep-learning/

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K


def getDistance(vectors):
    """
    Calculates euclidean distance between two vectors
    """
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def getOutputShape(shapes):
    """
    Returns euclidean distance output shape
    """
    shapeA, shapeB = shapes
    return (shapeA[0], 1)


def getContrastiveLoss(yTrue, yPred):
    '''
    Calculates Contrastive loss
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(yTrue * K.square(yPred) + (1 - yTrue) * K.square(K.maximum(margin - yPred, 0)))


def createSubNetwork(inputShape):
    '''
    Creates a base network for feature extraction.
    Three layer network utilizing euclidean distance for measuring distance similaritiy
    Batch normalize each layer, normalize the final feature vectors as it seems euclidean distances perform better with this
    '''

    # Initialize layer 1
    input = Input(shape=(inputShape,))
    dense1 = Dense(128)(input)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    # Initialize layer 2
    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = merge([relu1, bn2], mode='sum')
    relu2 = Activation('relu')(res2)

    # Initialize layer 3
    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = Merge(mode='sum')([relu2, bn3])
    relu3 = Activation('relu')(res3)

    # Merge everything and normalize one final time
    feats = merge([relu3, relu2, relu1], mode='concat')
    bn4 = BatchNormalization()(feats)

    return Model(input=input, output=bn4)


def getAccuracy(predictions, labels):
    '''
    Calculate classification accuracy through fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def createNetwork(inputShape):
    """
    Create siamese network with subnetworks
    """

    # Define network
    baseNetwork = createSubNetwork(inputShape)

    inputA = Input(shape=(inputShape,))
    inputB = Input(shape=(inputShape,))

    # As we reuse baseNetwork, the weights will be shared among both the branches
    processedA = baseNetwork(inputA)
    processedB = baseNetwork(inputB)

    # Evaluate distance
    distance = Lambda(getDistance, output_shape=getOutputShape)([processedA, processedB])
    return Model(input=[inputA, inputB], output=distance)
