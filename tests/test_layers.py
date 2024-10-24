import numpy as np
import pytest

from common.activation_functions import relu
from neural_networks.layers import ConnectedLayer


# test initialisation
def test_initialisation_with_weights():
    layer = ConnectedLayer(input_size=3, output_size=2)
    assert layer.weights is not None
    assert layer.biases is not None
    assert layer.weights.shape == (2, 3)
    assert layer.biases.shape == (2,)


def test_initialisation_without_weights():
    layer = ConnectedLayer(input_size=3, output_size=2, init_weights=False)
    assert layer.weights is None
    assert layer.biases is None


def test_initialisation_with_activation():
    layer = ConnectedLayer(input_size=3, output_size=2, activation="relu")

    assert layer.activation == relu


def test_initialisation_no_activation():
    layer = ConnectedLayer(input_size=3, output_size=2)

    assert layer.activation == None


# test forward pass


def test_forward_pass_without_activation():
    layer = ConnectedLayer(input_size=3, output_size=2)
    layer.input = np.array([1, 2, 3])

    # override weights to make calculation predictable
    layer.weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]])
    layer.biases = np.array([0, 0])

    output = layer.forward(layer.input)

    # expected output: W.x + b
    expected = np.array([1.4, 1.1])
    assert output == pytest.approx(expected)
    assert output.shape == (2,)


def test_forward_pass_with_activation():
    layer = ConnectedLayer(input_size=3, output_size=2, activation="sigmoid")
    layer.input = np.array([1, 2, 3])

    # override weights to make calculation predictable
    layer.weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]])
    layer.biases = np.array([1, 0])

    output = layer.forward(layer.input)

    # expected output: W.x + b
    expected = np.array([0.9168273, 0.7502601])
    assert output == pytest.approx(expected)
