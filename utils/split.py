import numpy as np
from numpy import ndarray
from typing import Tuple

from .datashuffling import DataShuffling

class Splitting:

    def __init__(self):
        pass

    def train_test_split(self, X: ndarray, Y: ndarray, test_size: float = 0.2) -> Tuple[ndarray, ndarray, ndarray, ndarray]:

        if not 0 < test_size < 1:
            raise ValueError(f"Test size must be between 0 and 1, you entered {test_size} which is not valid!")

        X, Y = DataShuffling().permute_data(X, Y)

        n_test = int(X.shape[0] * test_size)

        X_test, Y_test = X[:n_test], Y[:n_test]
        X_train, Y_train = X[n_test:], Y[n_test:]

        return X_train, Y_train, X_test, Y_test