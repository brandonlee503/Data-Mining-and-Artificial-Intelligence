import numpy as np
import pandas as pd

from test import run

def main():
    # Seperate data and class values
    train, test = getData()
    run(train[:,3:-1], train[:,-1])


# Parse CSV
def getData():
    training = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
    testing = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)

    return training.values, testing.values


if __name__ == '__main__':
    main()
