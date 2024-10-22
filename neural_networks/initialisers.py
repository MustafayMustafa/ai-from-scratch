from typing import Callable

import numpy as np


def get_initialiser(startegy: str) -> Callable[[int, int], np.ndarray]:
    initialiser_map = {
        "random_normal": random_normal,
        "random_uniform": random_uniform,
        "he_normal": he_normal,
        "he_uniform": he_uniform,
        "glorot_normal": glorot_normal,
        "glorot_uniform": glorot_uniform,
    }
    if startegy not in initialiser_map:
        raise ValueError("Initialisation strategy not defined")

    return initialiser_map[startegy]


def random_uniform(input_size, output_size):
    return np.random.uniform(-0.1, 0.1, size=(input_size, output_size))


def random_normal(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.1


def he_uniform(input_size, output_size):
    raise NotImplementedError


def he_normal(input_size, output_size):
    raise NotImplementedError


def glorot_uniform(input_size, output_size):
    raise NotImplementedError


def glorot_normal(input_size, output_size):
    raise NotImplementedError
