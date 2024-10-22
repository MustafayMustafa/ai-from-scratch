from typing import Callable
import numpy as np


def sigmoid(x):
    # clamp for numerical stability
    x = np.clip(x, -700, 700)

    # two formulations for numerical stability
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # subtract max value for numerical stability since it is invariant to constant shifts in the input
    exponentials = np.exp(x - np.max(x))
    return exponentials / np.sum(exponentials)


def tanh(x):
    # large values saturate to +-1 so can handle numerical stability like so
    return np.where(
        x > 20,
        1,
        np.where(x < -20, -1, (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))),
    )
