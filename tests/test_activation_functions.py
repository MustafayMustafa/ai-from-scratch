import pytest
import numpy as np
import numpy.testing as npt
from utils.activation_functions import sigmoid, relu, softmax, tanh


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(1) == pytest.approx(0.731058, rel=1e-5)
    assert sigmoid(-1) == pytest.approx(0.268941, rel=1e-5)

    # Edge case: very large positive input
    assert sigmoid(1000) == pytest.approx(1.0, rel=1e-5)
    # Edge case: very large negative input
    assert sigmoid(-1000) == pytest.approx(0.0, rel=1e-5)


def test_relu():
    input_data = np.array([1, 0, -1])
    expected_output = np.array([1, 0, 0])
    npt.assert_array_equal(relu(input_data), expected_output)


@pytest.fixture
def soft_max_positive_values():
    return np.array([2.0, 1.0, 0.1]), np.array([0.65900114, 0.24243297, 0.09856589])


@pytest.fixture
def soft_max_negative_values():
    return np.array([-2.0, -1.0, -0.1]), np.array([0.09611525, 0.26126834, 0.64261641])


@pytest.fixture
def soft_max_large_values():
    return np.array([1000, 1000, 1000]), np.array([1 / 3, 1 / 3, 1 / 3])


@pytest.fixture
def soft_max_zero_values():
    return np.array([0.0, 0.0, 0.0]), np.array([1 / 3, 1 / 3, 1 / 3])


def test_softmax_positive(soft_max_positive_values):
    input_data, expected_output = soft_max_positive_values
    result = softmax(input_data)
    npt.assert_allclose(result, expected_output, rtol=1e-6)


def test_softmax_negative(soft_max_negative_values):
    input_data, expected_output = soft_max_negative_values
    result = softmax(input_data)
    npt.assert_allclose(result, expected_output, rtol=1e-6)


def test_softmax_zero(soft_max_zero_values):
    input_data, expected_output = soft_max_zero_values
    result = softmax(input_data)
    npt.assert_allclose(result, expected_output, rtol=1e-6)


def test_softmax_large(soft_max_large_values):
    input_data, expected_output = soft_max_large_values
    result = softmax(input_data)
    npt.assert_allclose(result, expected_output, rtol=1e-6)


def test_tanh():
    assert tanh(0) == 0

    # Test for small positive and negative values
    assert tanh(1) == pytest.approx(np.tanh(1), rel=1e-6)
    assert tanh(-1) == pytest.approx(np.tanh(-1), rel=1e-6)

    # Test for large positive and negative values
    assert tanh(1000) == pytest.approx(np.tanh(1000), rel=1e-6)
    assert tanh(-1000) == pytest.approx(np.tanh(-1000), rel=1e-6)
