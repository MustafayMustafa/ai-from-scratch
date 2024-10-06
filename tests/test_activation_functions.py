import pytest
from utils.activation_functions import sigmoid, relu, softmax


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(1) == pytest.approx(0.731058, rel=1e-5)
    assert sigmoid(-1) == pytest.approx(0.268941, rel=1e-5)

    # Edge case: very large positive input
    assert sigmoid(1000) == pytest.approx(1.0, rel=1e-5)
    # Edge case: very large negative input
    assert sigmoid(-1000) == pytest.approx(0.0, rel=1e-5)


def test_relu():
    assert relu([1, 0, -1]) == [1, 0, 0]


@pytest.fixture
def basic_case():
    return [2.0, 1.0, 0.1], [0.65900114, 0.24243297, 0.09856589]


@pytest.fixture
def large_values_case():
    return [1000, 1000, 1000], [1 / 3, 1 / 3, 1 / 3]


@pytest.fixture
def negative_values_case():
    return [-1, -2, -3], [0.66524096, 0.24472847, 0.09003057]


@pytest.fixture
def zero_values_case():
    return [0.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]


def test_softmax_basic(basic_case):
    input_data, expected_output = basic_case
    softmax_result = softmax(input_data)
    assert all(
        [
            pytest.approx(result, rel=1e-6) == expected
            for result, expected in zip(softmax_result, expected_output)
        ]
    )
