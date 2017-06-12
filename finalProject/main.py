# Brandon Lee / John Miller
# References:
# http://www.erogol.com/duplicate-question-detection-deep-learning/

import os
import sys
import numpy as np
import pandas as pd

from siamese import *
from keras.optimizers import RMSprop, SGD

def main():
    # Seperate data and class values
    trainModel()


def trainModel():
    """
    A simple algorithm based off of word2vec
    Parse data in from CSV
    Convert mean word2vec representations of questions
    Train simple model for pairs and observe difference
    """

    # Read in values
    print 'Read in CSV values'
    training = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
    testing = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)

    # Encode questions to unicode
    print 'Encode training data to Unicode'
    training['question1'] = training['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    training['question2'] = training['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    print 'Encode testing data to Unicode'
    testing['question1'] = testing['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    testing['question2'] = testing['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    if os.path.exists('1_df.pkl'):
        print 'Reading existing pickle'
        df = pd.read_pickle('1_df.pkl')
    else:
        # Extract word2vec vectors, train GLOVE model
        import spacy
        print 'Loading spacy NLP'
        nlp = spacy.load('en')

        print 'Vectorizing 1'
        q1_vectors = [doc.vector for doc in nlp.pipe(training['question1'], n_threads=50)]
        q2_vectors = [doc.vector for doc in nlp.pipe(training['question2'], n_threads=50)]

        print 'Vectorizing 2'
        q1_vectors = np.array(q1_vectors)
        q2_vectors = np.array(q2_vectors)

        training['q1_features'] = list(q1_vectors)
        training['q2_features'] = list(q2_vectors)

        # Save features
        foo = open('1_df.pkl', 'w')
        pd.to_pickle(training, '1_df.pkl')
        foo.close()
        df = pd.read_pickle('1_df.pkl')

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
    optimizer = SGD(lr=0.1, momentum=0.8, nesterov=True, decay=0.004)
    net.compile(loss=getContrastiveLoss, optimizer=optimizer)

    for epoch in range(10):
        net.fit([xTrain[:,0,:], xTrain[:,1,:]], yTrain, validation_data=([xTest[:,0,:], xTest[:,1,:]], yTest), batch_size=128, nb_epoch=1, shuffle=True)

        # Calculate final accuracy of training and test sets
        prediction = net.predict([xTest[:,0,:], xTest[:,1,:]])
        testAccuracy = getAccuracy(prediction, yTest)

        print('*** Test Set Accuracy: %0.2f%%' % (100 * testAccuracy))


if __name__ == '__main__':
    main()
