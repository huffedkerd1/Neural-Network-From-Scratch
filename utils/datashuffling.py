'''
This file contain different methods to shuffle our data for unbiased results.
'''

import numpy as np
from numpy import ndarray
from typing import Tuple

class DataShuffling:
    def __init__(self):
        pass

    def permute_data(self, X: ndarray, Y: ndarray) -> Tuple[ndarray, ndarray]:

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        permutation = np.random.permutation(X.shape[0])

        return X[permutation], Y[permutation]