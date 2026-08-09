'''
This file is the most important file because it have the model Neural Network:

Now we see that what Neural Network class do?
    Taking the list of layers, which means how much layers this network have.
    Taking loss function class to calculate loss of our model.
    Setting seed.
'''

import numpy as np
from numpy import ndarray
from typing import List

# Our Blueprint classes for annotation only
from base import Layer
from base import Loss

# Our model starts from here
class NeuralNetwork:

    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):

        self.layers = layers
        self.loss = loss
        self.seed = seed 

        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    '''
    Taking the input but in batches.
    Store it.
    Pass input to first layer then calculate forward and feed to next layer.
    '''
    def forward(self, x_batch: ndarray) -> ndarray:

        x_out = x_batch

        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    '''
    Taking loss gradient to send it backward.
    Store it in grad.
    Then Calculate previous layer gradient and send it to backward.
    '''
    def backward(self, loss_grad: ndarray) -> None:

        grad = loss_grad

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    '''
    Taking batches of input and target values.

    Calculate forward pass.
    Calculate loss.
    Calculate backward pass.

    Return loss.
    '''
    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:

        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())

        return loss

    '''
    Gets the parameters for the network
    '''
    def params(self):

        for layer in self.layers:
            yield from layer.params

    '''
    Gets the gradient of loss with respect to each parameters of the network.
    '''
    def param_grads(self):

        for layer in self.layers:
            yield from layer.param_grads