from abc import ABC, abstractmethod

import numpy as np

import common.activation_functions as activation_functions
import neural_networks.initialisers as initialisers
from common.registry import get_callable


class BaseLayer(ABC):
    def __init__(
        self,
        input_size,
        output_size,
        strategy: str = "random_uniform",
        init_weights=True,
        activation=None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        self.activation = activation
        self.init_weights = init_weights
        if self.init_weights:
            self.initaliser = get_callable(strategy, initialisers)
            self.weights = self.initalise_weights(self.initaliser)
            self.biases = np.zeros(self.output_size)
        else:
            self.weights = None
            self.biases = None
        if self.activation:
            self.activation = get_callable(self.activation, activation_functions)

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self):
        pass

    def initalise_weights(self, initialiser):
        return initialiser(self.output_size, self.input_size)


class ConnectedLayer(BaseLayer):
    def forward(self, input_data):
        """Computes the linear transformation and activated value (if specified)
        z = W.x + b
        a = f(z)
        """
        self.input = input_data
        linear_transformation = np.dot(self.weights, self.input) + self.biases
        if self.activation:
            self.output = self.activation(linear_transformation)
        else:
            self.output = linear_transformation

        return self.output

    def backward(self):
        pass
