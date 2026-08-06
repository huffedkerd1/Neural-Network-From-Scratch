"""
This file have the blueprints of the whole process.

In this file we have 2 classes Operation, and ParamOperation. Both have different operations.
Every step of Neural Network have these operations.
"""

import numpy as np
from numpy import ndarray

"""
This class have the basic operation that almost every step have.

Inputs -> 1st Operation -> Output -> 2nd Operation -> Final Output
"""

class Operation:
    def __init__(self):
        pass

    def forward(self, input_: ndarray) -> ndarray:
        """
        Input -> function -> Output
            1. Storing Inputs.
            2. Calling Output function.
        """

        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        """
        Output Gradient -> derivative -> Input Gradient
            1. Checking shapes.
            2. Calling Input Gradient function.
        """

        assert self.output.shape == output_grad.shape

        self.input_grad = self._input_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape

        return self.input_grad

    def _output(self) -> ndarray:
        # Empty for now because every operation have own output.
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # Same as Output function
        raise NotImplementedError()

"""
This class have the blueprint of all operations that we make in parameters such as (W, b).
"""

class ParamOperation(Operation):

    # This class import the parent constructor.And have new attribute params. params -> W or b

    def __init__(self, params: ndarray) -> ndarray:
        super().__init__()
        self.params = params

    def backward(self, output_grad: ndarray) -> ndarray:

        """
            1. Checking shapes.
            2. Calling Input and Parameter Gradient.
        """

        assert self.output.shape == output_grad.shape

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_.shape == self.input_grad
        assert self.params == self.param_grad

        return self.input_grad, self.param_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()