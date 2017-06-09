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
    train, test = getData()
    print train[2]


def getData():
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
    print 'Encode to Unicode'
    training['question1'] = training['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    training['question2'] = training['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    #TODO: Encode Testing data

    if os.path.exists('1_df.pkl'):
        print 'Reading existing pickle'
        df = pd.read_pickle('1_df.pkl')
    else:
        # Extract word2vec vectors
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
        df = training

    # Check and see if distances are lookin good
    from scipy.spatial.distance import euclidean
    vec1 = df[df['qid1']==97]['q1_features'].values
    vec2 = df[df['qid2']==98]['q2_features'].values
    dist = euclidean(vec1[0], vec2[0])
    print("dist btw duplicate: %f" % (dist))

    vec1 = df[df['qid1']==91]['q1_features'].values
    vec2 = df[df['qid2']==92]['q2_features'].values
    dist = euclidean(vec1[0], vec2[0])
    print("dist btw non-duplicate: %f" % (dist))

    ### CREATE TRAINING DATA

    # Shuffle
    df = df.reindex(np.random.permutation(df.index))

    # set number of train and test instances
    num_train = int(df.shape[0] * 0.88)
    num_test = df.shape[0] - num_train
    print("Number of training pairs: %i"%(num_train))
    print("Number of testing pairs: %i"%(num_test))


    # init data data arrays
    X_train = np.zeros([num_train, 2, 300])
    X_test  = np.zeros([num_test, 2, 300])
    Y_train = np.zeros([num_train])
    Y_test = np.zeros([num_test])

    # format data
    b = [a[None,:] for a in list(df['q1_features'].values)]
    q1_feats = np.concatenate(b, axis=0)

    b = [a[None,:] for a in list(df['q2_features'].values)]
    q2_feats = np.concatenate(b, axis=0)

    # fill data arrays with features
    X_train[:,0,:] = q1_feats[:num_train]
    X_train[:,1,:] = q2_feats[:num_train]
    Y_train = df[:num_train]['is_duplicate'].values

    X_test[:,0,:] = q1_feats[num_train:]
    X_test[:,1,:] = q2_feats[num_train:]
    Y_test = df[num_train:]['is_duplicate'].values

    del b
    del q1_feats
    del q2_feats

    ### TRAIN MODEL
    net = create_network(300)

    # train
    optimizer = SGD(lr=0.1, momentum=0.8, nesterov=True, decay=0.004)
    #optimizer = RMSprop(lr=0.001)
    net.compile(loss=contrastive_loss, optimizer=optimizer)

    for epoch in range(10):
        net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
              validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
              batch_size=128, nb_epoch=1, shuffle=True)

        # compute final accuracy on training and test sets
        pred = net.predict([X_test[:,0,:], X_test[:,1,:]])
        te_acc = compute_accuracy(pred, Y_test)

        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return training.values, testing.values


if __name__ == '__main__':
    main()
