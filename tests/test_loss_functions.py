from common.loss_functions import (
    mean_absolute_error,
    mean_square_error,
    hinge_loss,
    binary_cross_entropy,
)
import numpy as np
import pytest


@pytest.fixture
def basic_case():
    truth = np.array([1, 1, 2, 0])
    predictions = np.array([1, 1, 4, 0])
    return truth, predictions


@pytest.fixture
def binary_case():
    truth = np.array([1, 1, 0, 0])
    predictions = np.array([0.8, 0.9, -0.5, -0.2])
    return truth, predictions


@pytest.fixture
def multi_class_case():
    pass


def test_mae(basic_case):
    truth, predictions = basic_case
    result = mean_absolute_error(truth, predictions)
    assert result == pytest.approx(0.5, rel=1e-5)


def test_mse(basic_case):
    truth, predictions = basic_case
    result = mean_square_error(truth, predictions)
    assert result == 1


def test_rmse(basic_case):
    truth, predictions = basic_case
    result = mean_square_error(truth, predictions)
    assert result == 1


def test_hinge_loss():
    y_true = np.array([-1, 1, 1, -1])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4


def test_binary_cross_entropy():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.3])

    expected_loss = (
        -(
            (y_true[0] * np.log(y_pred[0]) + (1 - y_true[0]) * np.log(1 - y_pred[0]))
            + (y_true[1] * np.log(y_pred[1]) + (1 - y_true[1]) * np.log(1 - y_pred[1]))
            + (y_true[2] * np.log(y_pred[2]) + (1 - y_true[2]) * np.log(1 - y_pred[2]))
            + (y_true[3] * np.log(y_pred[3]) + (1 - y_true[3]) * np.log(1 - y_pred[3]))
        )
        / 4
    )

    result = binary_cross_entropy(y_true, y_pred)
    assert result == pytest.approx(expected_loss, rel=1e-5)
