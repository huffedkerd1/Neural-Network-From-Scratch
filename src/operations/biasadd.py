import numpy as np
from numpy import ndarray

from base import ParamOperation

"""
This class adding bias in our weight multiply
y = XW + b
"""
class biasAdd(ParamOperation):

    def __init__(self, B: ndarray):
        """ Sending B parameter to ParamOperation class and checking its shape """
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self):
        # Calculate Output
        return self.input_ + self.params

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """ 
        Compute Input Gradient for bias add

        Input Gradient
            ∂L/∂X = 1 · ∂L/∂Y
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        """
        Now computing Bias Gradient

        Bias Gradient
            ∂L/∂B = 1 · ∂L/∂Y
        """
        param_grad = np.ones_like(self.params) * output_grad
        """
        This add bias row wise
        if after gradient over bias take this shape
        B = [1 2]
            [6 7]
            [8 2]

        we need to make it in one row 
        B = [1+6+8  2+7+2] -> Like This!
        """
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])