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


def getContrastiveLoss(y_true, y_pred):
    '''
    Calculates Contrastive loss
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def createSubNetwork(inputShape):
    '''
    Creates a base network for feature extraction.
    '''
    input = Input(shape=(inputShape,))
    dense1 = Dense(128)(input)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = merge([relu1, bn2], mode='sum')
    relu2 = Activation('relu')(res2)

    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = Merge(mode='sum')([relu2, bn3])
    relu3 = Activation('relu')(res3)

    feats = merge([relu3, relu2, relu1], mode='concat')
    bn4 = BatchNormalization()(feats)

    model = Model(input=input, output=bn4)

    return model


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

    input_a = Input(shape=(inputShape,))
    input_b = Input(shape=(inputShape,))

    # because we re-use the same instance `baseNetwork`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = baseNetwork(input_a)
    processed_b = baseNetwork(input_b)

    distance = Lambda(getDistance, output_shape=getOutputShape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)
    return model
