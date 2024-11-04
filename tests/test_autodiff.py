import numpy as np

import neural_networks.auto_diff as ad
from common.activation_functions import relu, sigmoid, softmax
from neural_networks.auto_diff import Tensor, exp, maximum, summation
import pytest


def test_addition():
    # test constants
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

    # test arrays
    a = Tensor(np.array([1, 1, 1]), track_gradient=True)
    b = Tensor(np.array([2, 2, 2]), track_gradient=True)
    z = a + b
    z.backward()

    assert np.array_equal(z.value, np.array([3, 3, 3]))
    assert np.array_equal(a.gradient, np.array([1, 1, 1]))
    assert np.array_equal(b.gradient, np.array([1, 1, 1]))


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

    # test arrays
    a = Tensor(np.array([4, 4, 4]), track_gradient=True)
    b = Tensor(np.array([1, 1, 1]), track_gradient=True)
    z = a - b
    z.backward()

    assert np.array_equal(z.value, np.array([3, 3, 3]))
    assert np.array_equal(a.gradient, np.array([1, 1, 1]))
    assert np.array_equal(b.gradient, np.array([-1, -1, -1]))


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

    # test arrays
    a = Tensor(np.array([4, 4, 4]), track_gradient=True)
    b = Tensor(np.array([2, 2, 2]), track_gradient=True)
    z = a * b
    z.backward()

    assert np.array_equal(z.value, np.array([8, 8, 8]))
    assert np.array_equal(a.gradient, np.array([2, 2, 2]))
    assert np.array_equal(b.gradient, np.array([4, 4, 4]))


def test_power_operation():
    base = Tensor(4.0, track_gradient=True)
    exponent = 3
    w = base**exponent
    w.backward()

    assert w.value == 64
    assert w.gradient == 1
    assert base.gradient == 48

    # test arrays
    base_array = Tensor(np.array([2, 3, 4]), track_gradient=True)
    result = base_array**exponent
    result.backward()

    assert np.array_equal(result.value, np.array([8, 27, 64]))
    assert np.array_equal(base_array.gradient, np.array([12, 27, 48]))


def test_division():
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

    # test array
    array_x = Tensor(np.array([1.0, 2.0, 3.0]), track_gradient=True)
    array_result = exp(array_x)
    array_result.backward()

    expected_array_value = np.exp(array_x.value)
    expected_array_gradient = expected_array_value

    assert np.allclose(array_result.value, expected_array_value)
    assert np.allclose(array_x.gradient, expected_array_gradient)


def test_negate_operation():
    x = Tensor(3.0, track_gradient=True)
    y = -x
    y.backward()

    assert y.value == -3.0
    assert np.isclose(x.gradient, -1.0)

    # test array
    array_x = Tensor(np.array([1.0, 2.0, 3.0]), track_gradient=True)
    array_y = -array_x
    array_y.backward()

    expected_array_value = np.array([-1.0, -2.0, -3.0])
    expected_array_gradient = -np.ones_like(array_x.value)

    assert np.array_equal(array_y.value, expected_array_value)
    assert np.array_equal(array_x.gradient, expected_array_gradient)


def test_sigmoid_operation():
    x = Tensor(0.0, track_gradient=True)
    result = sigmoid(x)
    result.backward()

    assert np.isclose(result.value, 1 / (1 + np.exp(-x.value)))
    assert np.isclose(x.gradient, result.value * (1 - result.value))

    array_x = Tensor(np.array([0.0, 2.0, -2.0]), track_gradient=True)
    array_result = sigmoid(array_x)
    array_result.backward()

    expected_values = 1 / (1 + np.exp(-array_x.value))
    expected_gradients = expected_values * (1 - expected_values)

    assert np.allclose(array_result.value, expected_values)
    assert np.allclose(array_x.gradient, expected_gradients)


def test_maximum():
    # test a>b
    a = Tensor(5.0, track_gradient=True)
    b = Tensor(3.0, track_gradient=True)
    result = maximum(a, b)
    result.backward()

    assert result.value == 5
    assert a.gradient == 1
    assert b.gradient == 0

    # test a<b
    a = Tensor(2.0, track_gradient=True)
    b = Tensor(3.0, track_gradient=True)
    result = maximum(a, b)
    result.backward()

    assert result.value == 3
    assert a.gradient == 0
    assert b.gradient == 1

    # test a==b
    a = Tensor(3.0, track_gradient=True)
    b = Tensor(3.0, track_gradient=True)
    result = maximum(a, b)
    result.backward()

    assert result.value == 3
    assert a.gradient == 0.5
    assert b.gradient == 0.5

    # test array
    a = Tensor(np.array([1.0, 5.0, 3.0]), track_gradient=True)
    b = Tensor(np.array([2.0, 4.0, 6.0]), track_gradient=True)
    result = maximum(a, b)
    result.backward()

    expected_values = np.maximum(a.value, b.value)
    assert np.array_equal(result.value, expected_values)
    assert np.array_equal(a.gradient, np.where(a.value > b.value, 1, 0))
    assert np.array_equal(b.gradient, np.where(b.value > a.value, 1, 0))


@pytest.mark.xfail
def test_relu():
    # x > 0
    a = Tensor(2.0, track_gradient=True)
    result = relu(a)
    result.backward()

    assert result.value == 2
    assert a.gradient == 1

    # x < 0
    b = Tensor(-1.0, track_gradient=True)
    result = relu(b)
    result.backward()

    assert result.value == 0
    assert b.gradient == 0

    # test array
    array_input = Tensor(np.array([-2.0, 0.0, 3.0]), track_gradient=True)
    array_result = relu(array_input)
    array_result.backward()

    expected_value = np.maximum(array_input.value, 0)
    expected_gradient = np.where(array_input.value > 0, 1, 0)

    assert np.array_equal(array_result.value, expected_value)
    assert np.array_equal(array_input.gradient, expected_gradient)


def test_reduce_max():
    x = Tensor(np.array([1, 4, 2]), track_gradient=True)
    result = ad.reduce_max(x)
    result.backward()

    assert result.value == 4
    assert result.gradient == 1
    expected_gradients = np.array([0.0, 1.0, 0.0])
    assert np.array_equal(x.gradient, expected_gradients)


def test_sum():
    x = Tensor(np.array([2, 3, 4]), track_gradient=True)
    result = summation(x)
    result.backward()

    assert result.value == 9.0
    assert np.isclose(result.gradient, 1.0)


@pytest.mark.xfail
def test_softmax():
    pass


@pytest.mark.xfail
def test_mean():
    a = Tensor(2.0, track_gradient=True)
    b = Tensor(4.0, track_gradient=True)
    c = Tensor(6.0, track_gradient=True)
    tensors = [a, b, c]
    result = ad.mean(tensors)

    expected_value = np.mean([2.0, 4.0, 6.0])
    assert np.isclose(result.value, expected_value)

    result.backward()
    expected_gradient = 1 / len(tensors)

    assert np.isclose(a.gradient, expected_gradient)
    assert np.isclose(b.gradient, expected_gradient)
    assert np.isclose(c.gradient, expected_gradient)


def test_sqrt():
    a = Tensor(4.0, track_gradient=True)
    result = ad.sqrt(a)

    expected_value = np.sqrt(4.0)
    assert np.isclose(result.value, expected_value)

    result.backward()
    expected_gradient = 0.5 / np.sqrt(4.0)
    assert np.isclose(a.gradient, expected_gradient)

    # test array
    array_tensor = Tensor(np.array([4.0, 9.0, 16.0]), track_gradient=True)
    result_array = ad.sqrt(array_tensor)

    expected_array_value = np.sqrt(array_tensor.value)
    assert np.allclose(result_array.value, expected_array_value)

    result_array.backward()
    expected_array_gradient = 0.5 / np.sqrt(array_tensor.value)

    assert np.allclose(array_tensor.gradient, expected_array_gradient)


def test_absolute():
    a = Tensor(5.0, track_gradient=True)
    result = ad.absolute(a)
    result.backward()

    assert result.value == 5.0
    assert np.isclose(a.gradient, 1.0)

    # Test for negative value
    b = Tensor(-3.0, track_gradient=True)
    result = ad.absolute(b)
    result.backward()

    assert result.value == 3.0
    assert np.isclose(b.gradient, -1.0)

    # Test for zero
    c = Tensor(0.0, track_gradient=True)
    result = ad.absolute(c)
    result.backward()

    assert result.value == 0.0
    assert np.isclose(c.gradient, 0.0)

    # test array
    array_tensor = Tensor(np.array([-4.0, 3.0, -2.0, 0.0]), track_gradient=True)
    result_array = ad.absolute(array_tensor)
    result_array.backward()

    expected_array_value = np.abs(array_tensor.value)
    expected_array_gradient = np.array([-1, 1, -1, 0])

    assert np.array_equal(result_array.value, expected_array_value)
    assert np.allclose(array_tensor.gradient, expected_array_gradient)


def test_log():
    # test psotiive value
    a = Tensor(2.0, track_gradient=True)
    result = ad.log(a)
    result.backward()

    assert np.isclose(result.value, np.log(2.0))
    assert np.isclose(a.gradient, 1.0 / a.value)

    # test close to 0
    c = Tensor(1e-10, track_gradient=True)
    result = ad.log(c)
    result.backward()

    assert np.isclose(result.value, np.log(1e-10))
    assert np.isclose(c.gradient, 1.0 / c.value)

    # test log(1)
    d = Tensor(1.0, track_gradient=True)
    result = ad.log(d)
    result.backward()

    assert np.isclose(result.value, np.log(1.0))
    assert np.isclose(d.gradient, 1.0 / d.value)

    # Test for an array of positive values
    array_a = Tensor(np.array([1.0, 2.0, 3.0]), track_gradient=True)
    array_result = ad.log(array_a)
    array_result.backward()

    expected_array_value = np.log(array_a.value)
    expected_array_gradient = 1.0 / array_a.value

    assert np.allclose(array_result.value, expected_array_value)
    assert np.allclose(array_a.gradient, expected_array_gradient)
