import numpy as np


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
