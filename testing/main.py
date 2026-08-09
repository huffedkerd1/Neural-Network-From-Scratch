'''
Let's test our first version of neural network.

By the way i am waiting for this so long.
I think this is my biggest Achivement,
Because i am for this line:
'nn = NeuralNetwork()' I love this for my personal Framework
'''

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layers import Dense
from src.losses import MeanSquaredError
from src.activations import Sigmoid
from models import NeuralNetwork

Layers = [
    Dense(neurons=1, activation=Sigmoid()),
    Dense(neurons=1, activation=Sigmoid())
]

nn = NeuralNetwork(layers=Layers, loss=MeanSquaredError(), seed=42)

X = np.random.randn(4, 2)

output = nn.forward(X)
print(f'Output: {output}')
print(f'Shape: {output.shape}')