import numpy as np
from neural_networks.auto_diff import Tensor, exp
from common.activation_functions import sigmoid, tanh


def test_addition():
    x = Tensor(1, track_gradient=True)
    y = Tensor(2, track_gradient=True)
    c = x + y
    w = c + y
    w.backward()

    assert w.value == 5
    assert w.gradient == 1
    assert c.gradient == 1
    assert x.gradient == 1
    assert y.gradient == 2


def test_subtraction():
    x = Tensor(1, track_gradient=True)
    y = Tensor(2, track_gradient=True)
    c = x - y
    w = c - y
    w.backward()

    assert w.value == -3
    assert w.gradient == 1
    assert c.gradient == 1
    assert x.gradient == 1
    assert y.gradient == -2


def test_multiplication():
    x = Tensor(3, track_gradient=True)
    y = Tensor(4, track_gradient=True)
    c = x * y
    w = c * y
    w.backward()

    assert w.value == 48
    assert w.gradient == 1
    assert c.gradient == 4
    assert x.gradient == 4 * 4
    assert y.gradient == 3 * 4 + 12


def test_power_operation():
    base = Tensor(4.0, track_gradient=True)
    exponent = 3
    w = base**exponent
    w.backward()

    assert w.value == 64
    assert w.gradient == 1
    assert base.gradient == 48


def test_division_operation():
    x = Tensor(6.0, track_gradient=True)
    y = Tensor(3.0, track_gradient=True)
    result = x / y
    result.backward()

    assert np.isclose(result.value, 2.0)
    assert np.isclose(x.gradient, 1 / y.value)
    assert np.isclose(y.gradient, -x.value / (y.value**2))


def test_exponential():
    x = Tensor(2.0, track_gradient=True)
    result = exp(x)
    result.backward()

    expected_value = np.exp(x.value)
    assert np.isclose(result.value, expected_value)
    assert np.isclose(x.gradient, expected_value)


def test_negate_operation():
    x = Tensor(3.0, track_gradient=True)
    y = -x
    y.backward()

    assert y.value == -3.0
    assert np.isclose(x.gradient, -1.0)


def test_sigmoid_operation():
    x = Tensor(0.0, track_gradient=True)
    result = sigmoid(x)
    result.backward()

    assert np.isclose(result.value, 1 / (1 + np.exp(-x.value)))
    assert np.isclose(x.gradient, result.value * (1 - result.value))
