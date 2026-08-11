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

from models import NeuralNetwork
from utils import Splitting
from src.layers import Dense
from src.activations import Sigmoid
from src.losses import MeanSquaredError
from src.optimizers import SGD
from src.train import Trainer

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

X_train, y_train, X_test, y_test = Splitting().train_test_split(X, y, test_size=0.25)

Layers = [
    Dense(neurons=4, activation=Sigmoid()),
    Dense(neurons=1, activation=Sigmoid())
]

nn = NeuralNetwork(layers=Layers, loss=MeanSquaredError(), seed=42)

output = nn.forward(X_train)

optimizer = SGD(lr=0.0001)
trainer = Trainer(nn, optimizer)
trainer.fit(X_train, y_train, X_test, y_test, epochs=10000, eval_every=100, batch_size=2, seed=42, restart=True )

output = nn.forward(X)
print(output)