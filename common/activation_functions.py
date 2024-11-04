from typing import Callable
import numpy as np
from neural_networks.auto_diff import Tensor, exp


def sigmoid(x):
    if isinstance(x, Tensor):
        constant_one = Tensor(1)
        return constant_one / (constant_one + exp(-x))
    else:
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
    if isinstance(x, Tensor):
        exponential = exp(x)
        negative_exponentail = exp(-x)
        return (exponential - negative_exponentail) / (
            exponential + negative_exponentail
        )
    else:

        if np.isscalar(x):
            if x > 20:
                return 1
            elif x < -20:
                return -1
            else:
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        x = np.asarray(x)
        result = np.zeros_like(x)

        # handle numerical stability
        result[x > 20] = 1
        result[x < -20] = -1

        mask = (x >= -20) & (x <= 20)
        result[mask] = (np.exp(x[mask]) - np.exp(-x[mask])) / (
            np.exp(x[mask]) + np.exp(-x[mask])
        )

        return result
