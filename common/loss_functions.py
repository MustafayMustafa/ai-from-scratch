import numpy as np
from neural_networks.auto_diff import Tensor
import neural_networks.auto_diff as ad


def mean_absolute_error(truth, prediction):
    if isinstance(truth, Tensor) and isinstance(prediction, Tensor):
        return ad.mean(ad.absolute(truth - prediction))
    else:
        return np.mean(np.abs(truth - prediction))


def mean_square_error(truth, prediction):
    if isinstance(truth, Tensor) and isinstance(prediction, Tensor):
        return ad.mean((truth - prediction) ** 2)
    else:
        return np.mean((truth - prediction) ** 2)


def root_mean_square_error(truth, prediction):
    if isinstance(truth, Tensor) and isinstance(prediction, Tensor):
        return ad.sqrt(ad.mean((truth - prediction) ** 2))
    else:
        return np.sqrt(np.mean((truth - prediction) ** 2))


def binary_cross_entropy(truth, prediction):
    # clip for numerical stability
    prediction = np.clip(prediction, 1e-15, 1 - 1e-15)

    if isinstance(truth, Tensor) and isinstance(prediction, Tensor):
        return -ad.mean(
            truth * ad.log(prediction) + (1 - truth) * ad.log(1 - prediction)
        )
    else:
        return -np.mean(
            truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction)
        )


def hinge_loss(truth, prediction):
    margin = truth * prediction
    losses = 1 - margin

    if isinstance(truth, Tensor) and isinstance(prediction, Tensor):
        return ad.mean(ad.maximum(0, losses))
    else:
        return np.mean(np.maximum(0, losses))
