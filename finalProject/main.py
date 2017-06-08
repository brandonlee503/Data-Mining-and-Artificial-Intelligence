import os
import sys
import numpy as np
import pandas as pd

def main():
    # Seperate data and class values
    train, test = getData()
    print train[2]


def getData():
    """
    Parse data in from CSV
    """

    # Read in values
    training = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
    testing = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)

    print "Encode to unicode"
    # Encode questions to unicode
    training['question1'] = training['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    training['question2'] = training['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    #TODO: Encode Testing data

    if os.path.exists('data/1_df.pkl'):
        df = pd.read_pickle('data/1_df.pkl')
    else:
        # Extract word2vec vectors
        import spacy
        print "Loading spacy NLP"
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
        pd.to_pickle(training, 'data/1_df.pkl')

    return training.values, testing.values


if __name__ == '__main__':
    main()
