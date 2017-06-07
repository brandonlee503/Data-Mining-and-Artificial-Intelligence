import numpy as np
import pandas as pd

def main():
    # Seperate data and class values
    train, test = getData()


# Parse CSV
def getData():
    training = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
    testing = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)

    return training.values, testing.values


if __name__ == '__main__':
    main()
