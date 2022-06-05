import os
from pathlib import Path
from typing import Tuple

import numpy as np

DIR = Path(os.path.dirname(os.path.realpath(__file__)))


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


def clear_cache(func_name: str):
    if func_name == "all" and os.path.isdir(DIR / "cache"):
        # remove all files in the cache directory
        for f in os.listdir(DIR / "cache"):
            os.remove(DIR / "cache" / f)
            return
    if os.path.isfile(DIR / "cache" / f"{func_name}.pkl"):
        os.remove(DIR / "cache" / f"{func_name}.pkl")
