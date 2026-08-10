'''
As we talk about there are different methods of Optimizers for
different dataset and different purposes.
'''

import numpy as np

from base import Optimizer

'''
Sochastic Gradient Descent is the most basic and powerful gradient descent technique.
For now we know that we use batch dataset, for batch Dataset SGD most powerful technique
for training.
'''

class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self):
        for (params, param_grads) in zip(self.net.params(), self.net.param_grads()):
            params -= self.lr * param_grads