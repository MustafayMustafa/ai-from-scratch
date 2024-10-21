import numpy as np


def mean_absolute_error(truth, prediction):
    return np.mean(np.abs(truth - prediction))


def mean_square_error(truth, prediction):
    return np.mean((truth - prediction) ** 2)


def root_mean_square_error(truth, prediction):
    return np.sqrt(np.mean((truth - prediction) ** 2))


def binary_cross_entropy(truth, prediction):
    # clip for numerical stability
    prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
    return -np.mean(truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))


def hinge_loss(truth, prediction):
    margin = truth * prediction
    losses = 1 - margin
    return np.average(np.maximum(0, losses))
