import numpy as np

import neural_networks.auto_diff as ad
from common.activation_functions import relu, sigmoid, softmax, tanh
from common.loss_functions import (
    binary_cross_entropy,
    hinge_loss,
    mean_absolute_error,
    mean_square_error,
    root_mean_square_error,
)
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

    # test arrays
    x = Tensor(np.array([6, 8, 10]), track_gradient=True)
    y = Tensor(np.array([3, 4, 5]), track_gradient=True)
    result = x / y
    result.backward()

    expected_value = np.array([6.0 / 3.0, 8.0 / 4.0, 10.0 / 5.0])
    assert np.allclose(result.value, expected_value)

    expected_x_gradient = 1 / y.value
    expected_y_gradient = -x.value / (y.value**2)
    assert np.allclose(x.gradient, expected_x_gradient)
    assert np.allclose(y.gradient, expected_y_gradient)
    assert np.array_equal(result.gradient, np.array([1, 1, 1]))


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
    assert np.array_equal(array_result.value, expected_value)
    assert np.array_equal(array_input.gradient, np.array([0, 0, 1]))
    assert np.array_equal(array_result.gradient, np.array([1, 1, 1]))


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


def test_softmax():
    x_values = np.array([1.0, 3.0, 3.0])
    x = ad.Tensor(x_values, track_gradient=True)

    max_value = np.max(x_values)
    exp_values = np.exp(x_values - max_value)
    expected_output = exp_values / np.sum(exp_values)
    result = softmax(x)

    assert np.allclose(result.value, expected_output)
    assert np.isclose(np.sum(result.value), 1.0), "Softmax output does not sum to 1"

    result.backward()

    s = expected_output.reshape(-1, 1)
    expected_jacobian = np.diagflat(s) - np.dot(s, s.T)
    expected_gradient = np.dot(expected_jacobian, np.ones_like(x_values))
    assert np.allclose(x.gradient, expected_gradient)


def test_mean():
    x = Tensor(np.array([1, 2, 3]), track_gradient=True)
    # a = Tensor(2.0, track_gradient=True)
    # b = Tensor(4.0, track_gradient=True)
    # c = Tensor(6.0, track_gradient=True)
    result = ad.mean(x)

    expected_value = np.mean([1, 2, 3])
    assert np.isclose(result.value, expected_value)

    result.backward()
    expected_gradients = np.array([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(x.gradient, expected_gradients)
    assert result.gradient == 1


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


def test_tanh():
    x_values = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    x = ad.Tensor(x_values, track_gradient=True)
    result = tanh(x)

    expected_output = np.tanh(x_values)
    assert np.allclose(result.value, expected_output)

    result.backward()
    expected_gradient = 1 - np.square(expected_output)
    assert np.allclose(x.gradient, expected_gradient)


def test_mean_absolute_error():
    truth_values = np.array([3.0, -0.5, 2.0, 7.0])
    prediction_values = np.array([2.5, 0.0, 2.0, 8.0])
    truth = ad.Tensor(truth_values, track_gradient=True)
    prediction = ad.Tensor(prediction_values, track_gradient=True)
    mae_result = mean_absolute_error(truth, prediction)

    expected_mae = np.mean(np.abs(truth_values - prediction_values))
    assert np.isclose(mae_result.value, expected_mae)

    mae_result.backward()
    N = truth_values.size
    expected_truth_gradient = np.sign(truth_values - prediction_values) / N
    expected_prediction_gradient = -np.sign(truth_values - prediction_values) / N
    assert np.allclose(truth.gradient, expected_truth_gradient)
    assert np.allclose(prediction.gradient, expected_prediction_gradient)


def test_mean_square_error():
    truth_values = np.array([3.0, -0.5, 2.0, 7.0])
    prediction_values = np.array([2.5, 0.0, 2.0, 8.0])
    truth = ad.Tensor(truth_values, track_gradient=True)
    prediction = ad.Tensor(prediction_values, track_gradient=True)
    mse_result = mean_square_error(truth, prediction)

    expected_mse = np.mean((truth_values - prediction_values) ** 2)
    assert np.isclose(mse_result.value, expected_mse)

    mse_result.backward()
    N = truth_values.size
    expected_truth_gradient = 2 * (truth_values - prediction_values) / N
    expected_prediction_gradient = -2 * (truth_values - prediction_values) / N
    assert np.allclose(truth.gradient, expected_truth_gradient)
    assert np.allclose(prediction.gradient, expected_prediction_gradient)


def test_root_mean_square_error():
    truth_values = np.array([3.0, -0.5, 2.0, 7.0])
    prediction_values = np.array([2.5, 0.0, 2.0, 8.0])
    truth = ad.Tensor(truth_values, track_gradient=True)
    prediction = ad.Tensor(prediction_values, track_gradient=True)
    rmse_result = root_mean_square_error(truth, prediction)

    expected_rmse = np.sqrt(np.mean((truth_values - prediction_values) ** 2))
    assert np.isclose(rmse_result.value, expected_rmse)

    rmse_result.backward()
    N = truth_values.size
    expected_truth_gradient = (truth_values - prediction_values) / (N * expected_rmse)
    expected_prediction_gradient = -(truth_values - prediction_values) / (
        N * expected_rmse
    )
    assert np.allclose(truth.gradient, expected_truth_gradient)
    assert np.allclose(prediction.gradient, expected_prediction_gradient)


def test_hinge_loss():
    truth_values = np.array([1, -1, 1, -1])
    prediction_values = np.array([0.8, -0.5, -1.2, 0.3])
    truth = ad.Tensor(truth_values, track_gradient=False)
    prediction = ad.Tensor(prediction_values, track_gradient=True)
    hinge_loss_result = hinge_loss(truth, prediction)

    margin = truth_values * prediction_values
    expected_hinge_loss = np.mean(np.maximum(0, 1 - margin))
    assert np.isclose(hinge_loss_result.value, expected_hinge_loss)

    hinge_loss_result.backward()
    N = truth_values.size
    expected_prediction_gradient = np.where((1 - margin) > 0, -truth_values / N, 0)
    assert np.allclose(prediction.gradient, expected_prediction_gradient)


def test_binary_cross_entropy():
    truth_values = np.array([1, 0, 1, 1])
    prediction_values = np.array([0.9, 0.2, 0.8, 0.7])
    prediction_values = np.clip(prediction_values, 1e-15, 1 - 1e-15)
    truth = ad.Tensor(truth_values, track_gradient=True)
    prediction = ad.Tensor(prediction_values, track_gradient=True)
    bce_result = binary_cross_entropy(truth, prediction)

    expected_bce = -np.mean(
        truth_values * np.log(prediction_values)
        + (1 - truth_values) * np.log(1 - prediction_values)
    )
    assert np.isclose(bce_result.value, expected_bce)

    bce_result.backward()
    # N = truth_values.size
    # expected_prediction_gradient = (
    #     -(
    #         truth_values / prediction_values
    #         - (1 - truth_values) / (1 - prediction_values)
    #     )
    #     / N
    # )

    # assert np.allclose(prediction.gradient, expected_prediction_gradient)
