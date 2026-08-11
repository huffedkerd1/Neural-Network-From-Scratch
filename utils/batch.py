import numpy as np
from numpy import ndarray
from typing import Tuple

class BatchData:

    def __init__(self):
        pass

    def generate_batch(self, X: ndarray, Y:ndarray, start: int = 0, batch_size: int = 32):

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        
        for start in range(0, X.shape[0], batch_size):

            X_batch = X[start:start + batch_size]
            Y_batch = Y[start:start + batch_size]

            yield X_batch, Y_batch