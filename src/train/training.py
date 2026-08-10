import numpy as np
from numpy import ndarray

from models import NeuralNetwork
from base import Optimizer

from utils import DataShuffling
from utils import BatchData

class Trainer:

    def __init__(self, net: NeuralNetwork, optimi: Optimizer):

        self.net = net
        self.optimi = optimi
        setattr(self.optimi, "net", self.net)

    def fit(
            self,
            X_train: ndarray, y_train: ndarray, 
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True
            ) -> None:

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

        for e in range(epochs):

            X_train, y_train = DataShuffling().permute_data(X_train, y_train)
            batchdata = BatchData().generate_batch(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batchdata):
                self.net.train_batch(X_batch, y_batch)
                self.optimi.step()

            if (e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                
                print(f'Loss after {e+1} epochs is {loss:.3f}')