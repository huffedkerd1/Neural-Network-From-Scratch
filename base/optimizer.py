'''
Let's start most important part of Neural Network:

The Neural Network feed input data(X) to each layer to calculate
predictions then we compare our input data(X) to actual values(y)
to calculate how wrong our model is -> Loss function, 
then calculate the derivative of loss function with respect to
parameters in each layer.

After calculating dervatives or gradients we need to correct our model.
There are many methods to correct our model, and the concept to update
our parameters we called Optimizers.

There are different types of Optimizers.

We choose Optimizers depends on our dataset.

'''

'''
The blueprint class of Optimizers.
'''
import numpy as np

class Optimizer:

    # Taking and Storing learning rate in Optimizer class, Which is later help in training our model.
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        pass