from neural_networks import initialisers
import pytest


def test_get_initaliser():

    assert initialisers.get_initialiser("random_uniform") == initialisers.random_uniform
    assert initialisers.get_initialiser("random_normal") == initialisers.random_normal


def test_get_initaliser_unknown():
    with pytest.raises(ValueError, match="Initialisation strategy not defined"):
        initialisers.get_initialiser("dummy")
