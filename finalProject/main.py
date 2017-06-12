# Brandon Lee / John Miller
# References:
# http://www.erogol.com/duplicate-question-detection-deep-learning/

import os
import sys
import numpy as np
import pandas as pd
import json

from siamese import *
from keras.optimizers import RMSprop, SGD, Adam

from keras.models import model_from_json

def main():
    # Seperate data and class values
    # trainModel()
    testModel()


def trainModel():
    """
    A simple algorithm based off of word2vec
    Parse data in from CSV
    Convert mean word2vec representations of questions
    Train simple model for pairs and observe difference
    """

    # Read in values
    print 'Training - Read in CSV values'
    trainingDF = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)

    # Encode questions to unicode
    print 'Training - Encode training data to Unicode'
    trainingDF['question1'] = trainingDF['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    trainingDF['question2'] = trainingDF['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    # Training GLOVE Model
    if os.path.exists('trainModelGLOVE.pkl'):
        print 'Training - Reading existing training pickle'
        df = pd.read_pickle('trainModelGLOVE.pkl')
    else:
        # Extract word2vec vectors, train GLOVE model
        import spacy
        print 'Training - Loading spacy NLP'
        nlp = spacy.load('en')

        print 'Training - Vectorizing 1'
        q1_vectors = [doc.vector for doc in nlp.pipe(trainingDF['question1'], n_threads=50)]
        q2_vectors = [doc.vector for doc in nlp.pipe(trainingDF['question2'], n_threads=50)]

        print 'Training - Vectorizing 2'
        q1_vectors = np.array(q1_vectors)
        q2_vectors = np.array(q2_vectors)

        trainingDF['q1_features'] = list(q1_vectors)
        trainingDF['q2_features'] = list(q2_vectors)

        # Save features
        foo = open('trainModelGLOVE.pkl', 'w')
        pd.to_pickle(trainingDF, 'trainModelGLOVE.pkl')
        foo.close()
        df = pd.read_pickle('trainModelGLOVE.pkl')

    ### CREATE TRAINING DATA

    # Shuffle
    df = df.reindex(np.random.permutation(df.index))

    # Get the total number of training and testing instances
    totalTraining = int(df.shape[0] * 0.88)
    totalTesting = df.shape[0] - totalTraining
    print("Total training pairs: %i"%(totalTraining))
    print("Total testing pairs: %i"%(totalTesting))

    # Initialize data arrays
    xTrain = np.zeros([totalTraining, 2, 300])
    xTest  = np.zeros([totalTesting, 2, 300])
    yTrain = np.zeros([totalTraining])
    yTest = np.zeros([totalTesting])

    # Refactor data
    b = [a[None,:] for a in list(df['q1_features'].values)]
    q1_features = np.concatenate(b, axis=0)

    b = [a[None,:] for a in list(df['q2_features'].values)]
    q2_features = np.concatenate(b, axis=0)

    # Fill the data arrays with features
    xTrain[:,0,:] = q1_features[:totalTraining]
    xTrain[:,1,:] = q2_features[:totalTraining]
    yTrain = df[:totalTraining]['is_duplicate'].values

    xTest[:,0,:] = q1_features[totalTraining:]
    xTest[:,1,:] = q2_features[totalTraining:]
    yTest = df[totalTraining:]['is_duplicate'].values

    del b
    del q1_features
    del q2_features

    ### TRAIN MODEL
    net = createNetwork(300)

    # Perform actual training with siamese network
    optimizer = Adam(lr=0.001)

    if os.path.exists('net_weights.h5'):
        net.load_weights('net_weights.h5')
        net.compile(loss=getContrastiveLoss, optimizer=optimizer)

        # Calculate final accuracy of training and test sets
        prediction = net.predict([xTest[:,0,:], xTest[:,1,:]])
        testAccuracy = getAccuracy(prediction, yTest)

        print('*** Test Set Accuracy: %0.2f%%' % (100 * testAccuracy))
    else:
        net.compile(loss=getContrastiveLoss, optimizer=optimizer)

        for epoch in range(6):
            print('Beginning epoch {0}'.format(epoch))
            net.fit([xTrain[:,0,:], xTrain[:,1,:]], yTrain, validation_data=([xTest[:,0,:], xTest[:,1,:]], yTest), batch_size=128, epochs=1, shuffle=True)

            # Calculate final accuracy of training and test sets
            prediction = net.predict([xTest[:,0,:], xTest[:,1,:]])
            testAccuracy = getAccuracy(prediction, yTest)

            print('*** Test Set Accuracy: %0.2f%%' % (100 * testAccuracy))

        net.save_weights('net_weights.h5')


def testModel():
    """
    Build GLOVE Model for testing data
    """

    print 'Testing - Read in CSV values'
    testingDF = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)

    print 'Testing - Encode testing data to Unicode'
    testingDF['question1'] = testingDF['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    testingDF['question2'] = testingDF['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    # Split data into subsection for speed
    testingDF = testingDF[:10]
    print testingDF

    # Testing GLOVE Model
    if os.path.exists('testModelGLOVE.pkl'):
        print 'Testing - Reading existing pickle'
        df = pd.read_pickle('testModelGLOVE.pkl')
    else:
        # Extract word2vec vectors, train GLOVE model
        import spacy
        print 'Testing - Loading spacy NLP'
        nlp = spacy.load('en')

        print 'Testing - Vectorizing 1'
        q1_vectors = [doc.vector for doc in nlp.pipe(testingDF['question1'], n_threads=50)]
        q2_vectors = [doc.vector for doc in nlp.pipe(testingDF['question2'], n_threads=50)]

        print 'Testing - Vectorizing 2'
        q1_vectors = np.array(q1_vectors)
        q2_vectors = np.array(q2_vectors)

        testingDF['q1_features'] = list(q1_vectors)
        testingDF['q2_features'] = list(q2_vectors)

        # Save features
        foo = open('testModelGLOVE.pkl', 'w')
        pd.to_pickle(testingDF, 'testModelGLOVE.pkl')
        foo.close()
        df = pd.read_pickle('testModelGLOVE.pkl')

if __name__ == '__main__':
    main()
