'''
Let's test our first version of neural network.

By the way i am waiting for this so long.
I think this is my biggest Achivement,
Because i am waiting for this line:
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
from src.optimizers import SGD
from src.train import Trainer

X_train = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.3, 0.2],
    [0.2, 0.3],
    [0.8, 0.9],
    [0.9, 0.8],
    [0.7, 0.9],
    [0.9, 0.7]
])

y_train = np.array([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1]
])

X_test = np.array([
    [0.15, 0.25],
    [0.85, 0.85],
    [0.25, 0.15],
    [0.75, 0.80]
])

y_test = np.array([
    [0],
    [1],
    [0],
    [1]
])

Layers = [
    Dense(neurons=10, activation=Sigmoid()),
    Dense(neurons=7, activation=Sigmoid()),
    Dense(neurons=5, activation=Sigmoid()),
    Dense(neurons=1, activation=Sigmoid()),
    Dense(neurons=1, activation=Sigmoid())
]

nn = NeuralNetwork(layers=Layers, loss=MeanSquaredError(), seed=42)
output = nn.forward(X_train)

<<<<<<< HEAD
optimizer = SGD(lr=0.01)
train = Trainer(nn, optimizer)
train.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10, batch_size=5, seed=42, restart=True) 
=======
X = np.random.randn(4, 2)

output = nn.forward(X)
print(f'Output: {output}')
print(f'Shape: {output.shape}')
>>>>>>> 438d911c4c792b71834b6407fd575646afdbf1a7
