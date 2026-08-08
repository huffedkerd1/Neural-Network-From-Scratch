"""
In this file we are going to see that how we multiply inputs with weights.

In this file we have a class name WeightMultiply which is inherit by our ParamOperation class.
This class handle weight multiply and its derivative.
"""

import numpy as np
from numpy import ndarray

# Importing our ParamOperation class.
from base import ParamOperation


class WeightMultiply(ParamOperation):

    def __init__(self, W: ndarray):
        """As we know that in ParamOperation we have an attribute params now we pass weights(W) as a param"""
        assert self.input_.shape[1] == W.shape[0]
        super().__init__(W)

    def _output(self) -> ndarray:
        # Calculating Weight Multiply => XW
        return np.dot(self.input_, self.params)

    # Gradient help us in when go backward
    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        In this process we calculate Input Gradient which is:

        Input Gradient
            ∂L/∂X = Wᵀ · ∂L/∂Y

            For now we dont know what is 'derivative of loss with respect to y'
            We calculate in next operations
        """
        return np.dot(output_grad, self.params.T)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Weight Gradient
            ∂L/∂W = Xᵀ · ∂L/∂Y
        """
        return np.dot(self.input_.T, output_grad)
