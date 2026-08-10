'''
Now we apply our loss formula and its derivative.

In this file we are gonna apply Mean Squared Error(MSE) loss.
If you check out our documentation you know MSE formula.
'''

import numpy as np
from numpy import ndarray

# Importing our Loss class
from base import Loss

class MeanSquaredError(Loss):

    def __init__(self):
        super().__init__()

    '''
    Calculating loss formula -> MSE.
    '''
    def _output(self) -> float:

        return np.mean(np.pow(self.predictions - self.target, 2))

    '''
    Calculating loss derivative -> MSE.
    '''
    def _input_grad(self) -> ndarray:

        return 2.0 * (self.predictions - self.target) / self.predictions.size