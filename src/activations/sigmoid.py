"""
Now we define our first activation function which add non-linearity in every single layer.
This function is class Sigmoid.
"""

import numpy as np
from numpy import ndarray

from base import Operation

class Sigmoid(Operation):

    # Inherit constructor from parent class
    def __init__(self) -> None:
        super().__init__()

    # Apply the formula of sigmoid function
    def _output(self) -> ndarray:
        output_ = np.empty_like(self.input_, dtype=float)

        positive = self.input_ >= 0
        output_[positive] = 1.0 / (1.0 + np.exp(-self.input_[positive]))

        exp_input = np.exp(self.input_[~positive])
        output_[~positive] = exp_input / (1.0 + exp_input)

        return output_

    # Caluculate the derivative of sigmoid function
    def _input_grad(self, output_grad: ndarray) -> ndarray:

        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad