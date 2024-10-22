import sys

import pytest
from common.registry import get_definitions, get_callable


def mock_function():
    return 1


test_module = sys.modules[__name__]


def test_get_definitions():
    result = get_definitions(test_module)

    assert "mock_function" in result
    assert callable(result["mock_function"])


def test_get_callable():
    func = get_callable("mock_function", test_module)
    assert func == mock_function
    assert func() == 1


def test_get_callable_undefined():
    with pytest.raises(
        ValueError, match="Function 'dummy' not found in module 'test_registry'"
    ):
        get_callable("dummy", test_module)
