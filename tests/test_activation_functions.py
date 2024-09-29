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


@pytest.mark.xfail(reason="Function not implemented yet")
def test_relu():
    pass


@pytest.mark.xfail(reason="Function not implemented yet")
def test_softmax():
    pass
