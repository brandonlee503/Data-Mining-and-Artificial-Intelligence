import numpy as np

class LR:
    def __init__(self):
        pass

    @staticmethod
    def computePrediction(row, w):
        return 1.0 / (1.0 + np.exp(-1.0 * (np.dot(w, row))))

    def computeCoefficient(self, x, y, w):
        d = np.zeros(w.shape[0])

        for row in range(len(x)):
            yHat = self.computePrediction(x[row], w)
            error = float(y[row]) - yHat
            d = np.add(d, np.multiply(error, x[row]))
        return d

    def testData(self, w, x, y):
        correct = 0
        for i, row in enumerate(x):
            prediction = self.computePrediction(row, w)
            if round(prediction) == y[i]:
                correct = correct + 1

        return float(correct) / float(y.shape[0])

    def computeCoefficientReg(self, x, y, w):
        d = np.zeros(w.shape[0])

        for row in range(len(x)):
            yHat = self.computePrediction(x[row], w)
            error = float(y[row]) - yHat
            d = np.add(d, np.multiply(error, x[row]))
        return d
