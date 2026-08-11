"""
Now let's design blueprint of layer.

In this layer we see that how single layer work and how maths work.
"""

import numpy as np
from numpy import ndarray
from typing import List

from .operation import ParamOperation

# Making the layer class to inheriate in different types of layers
class Layer:

    def __init__(self, neurons: int):
        self.neurons = neurons # Number of Neurons
        self.first = True # Is this layer first time used means it not have any inputs
        self.params: List[ndarray] = [] # weigths and bias
        self.param_grads: List[ndarray] = [] # derivative of weigths and bias
        self.operations: List[ndarray] = [] # Operations include in layer

    def _setup_layer(self, input_: ndarray) -> None:
        '''Implemented for each layer'''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        # Checking is this first layer, if it is first layer then send input in setup layer method to generate initial weights and biases
        # And set first to false
        if self.first:
            self._setup_layer(input_)
            self.first = False

        # Storing input
        self.input_ = input_

        # Opening Operations list and apply forward operation
        for operation in self.operations:
            input_ = operation.forward(input_)

        # Return output
        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        # Checking output and output gradients shape
        assert self.output.shape == output_grad.shape

        # In forward pass we go forward and In backward we go backward
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        # Return input_grad and calling parameters gradient
        input_grad = output_grad
        self.param_grad()

        return input_grad

    def param_grad(self) -> None:
        # Storing parameters gradient 
        self.param_grads = []
        '''
        Checking is any operation have parameters which is a subclass of ParamOperation, then calculate its gradient
        '''
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def param(self) -> None:
        # Storing parameters
        self.params = []
        '''
        Checking is any operation have parametes which is a subclass of ParamOperation, then take out the parameters
        '''
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.params)
