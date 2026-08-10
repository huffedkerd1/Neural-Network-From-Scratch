import numpy as np
from numpy import ndarray
from typing import Tuple

class BatchData:

    def __init__(self):
        pass

    def generate_batch(self, X: ndarray, Y:ndarray, start: int = 0, batch_size: int = 32):

        if start + batch_size > X.shape[0]:
            batch_size = X.shape[0] - start

        X_batch = X[start: start + batch_size]
        Y_batch = Y[start: start + batch_size]

        return X_batch, Y_batch