from typing import Tuple
import numpy as np


def train_test_split(
    X: np.ndarray, Y: np.ndarray, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inds = np.random.permutation(len(X))
    X = X[inds]
    Y = Y[inds]

    test_sz = int(len(X) * test_size)
    trainX, testX = X[test_sz:], X[:test_sz]
    trainY, testY = Y[test_sz:], Y[:test_sz]
    return trainX, trainY, testX, testY
