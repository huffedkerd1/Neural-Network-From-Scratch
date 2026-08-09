'''
Let's do the last step:

As we know that forward pass takes inputs passes from multiple hidden layers
and the pass to output layer to generate predictions.

But how we know that how wrong model predict the values?

Note: 'Predictions are approximately equal to target values but not exact equal or target.'

The answer is Loss function, loss function compare predictions and target values to check
how wrong our model is.

After calculating loss values we need to correct our model for more accurate predictions.
Now the question is how we correct our model?
The answer is in the docs folder.
you need to check it.

Loss function do some operations like:
    Forward:

    Taking predictions and target values as inputs.
    Then store predictions and target values for different purpose.
    And last calculate loss values.

    Backward:

    Calculating and storing input gradients to send it backward.
'''

import numpy as np
from numpy import ndarray

class Loss:

    def __init__(self):
        pass

    def forward(self, predictions: ndarray, target: ndarray) -> float:

        assert predictions.shape == target.shape

        self.predictions = predictions
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:

        self.input_grad = self._input_grad()

        assert self.predictions.shape == self.input_grad.shape

        return self.input_grad

    '''
    This output function is empty for now beacause we have many loss formulas,
    and these formulas are written in this function.
    '''
    def _output(self) -> float:
        raise NotImplementedError()

    '''
    And also if we have multiple loss formulas, so each formula have different derivatives.
    Which is written in this function.
    '''
    def _input_grad(self) -> ndarray:
        raise NotImplementedError()