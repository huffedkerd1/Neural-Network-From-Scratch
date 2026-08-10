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
        permutation = np.random.permutation(X.shape[0])

        return X[permutation], Y[permutation]