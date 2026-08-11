'''
In this file we are gonna create Dense Layer.

Dense Layer also known as 'Fully connected layer' 
which means every single layer fully connected to next layer.
'''

import numpy as np
from numpy import ndarray

# Importing all classes except ParamOperation class
from base import Layer, Operation
from ..operations import WeightMultiply, BiasAdd
from ..activations import Sigmoid

class Dense(Layer):

    def __init__(self, neurons: int, activation: Operation = None) -> None:
        super().__init__(neurons)
        self.activation = activation if activation is not None else Sigmoid()

    def _setup_layer(self, input_: ndarray) -> None:
        
        # Creating empty params list. This list contain weights and biases.
        self.params = []

        self.params.append(np.random.randn(input_.shape[1], self.neurons) * np.sqrt(2 / input_.shape[1])) # Weights
        self.params.append(np.zeros((1, self.neurons))) # Bias

        # Setting all operations in one operation list. We already use in Layer class. But operations define here.
        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation
        ]

        return None
