from typing import Callable
import numpy as np
from neural_networks import auto_diff as ad


def sigmoid(x):
    if isinstance(x, ad.Tensor):
        constant_one = ad.Tensor(1)
        return constant_one / (constant_one + ad.exp(-x))
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
    if isinstance(x, ad.Tensor):
        return ad.relu(x)
    else:
        return np.maximum(0, x)


def softmax(x):
    if isinstance(x, ad.Tensor):
        return ad.softmax(x)

    else:
        # subtract max value for numerical stability since it is invariant to constant shifts in the input
        exponentials = np.exp(x - np.max(x))
        return exponentials / np.sum(exponentials)


def tanh(x):
    if isinstance(x, ad.Tensor):
        return ad.tanh(x)
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
