import numpy as np


def add(a, b):
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(a.value + b.value, track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient
        if b.track_gradient:
            b.gradient = (b.gradient or 0) + output_tensor.gradient

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def subtract(a, b):
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(a.value - b.value, track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient
        if b.track_gradient:
            b.gradient = (b.gradient or 0) - output_tensor.gradient

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def multiply(a, b):
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(a.value * b.value, track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient * b.value
        if b.track_gradient:
            b.gradient = (b.gradient or 0) + output_tensor.gradient * a.value

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def power(base, exponent):
    track_gradient = base.track_gradient
    result = Tensor(base.value**exponent, track_gradient)

    def _backward(output_tensor):
        if base.track_gradient:
            base.gradient = (base.gradient or 0) + output_tensor.gradient * exponent * (
                base.value ** (exponent - 1)
            )

    result.backward_function = _backward
    result.parents = [base]
    return result


def exp(exponent):
    track_gradient = exponent.track_gradient
    result = Tensor(np.exp(exponent.value), track_gradient)

    def _backward(output_tensor):
        if exponent.track_gradient:
            exponent.gradient = (
                exponent.gradient or 0
            ) + output_tensor.gradient * result.value

    result.backward_function = _backward
    result.parents = [exponent]
    return result


def negate(x):
    result = Tensor(-x.value, x.track_gradient)

    def _backward(output_tensor):
        if x.track_gradient:
            x.gradient = (x.gradient or 0) - output_tensor.gradient

    result.backward_function = _backward
    result.parents = [x]

    return result


def maximum(a, b):
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(max(a.value, b.value), track_gradient)

    def _backward(output_tensor):
        if a.track_gradient and a.value > b.value:
            a.gradient = (a.gradient or 0) + output_tensor.gradient
        if b.track_gradient and a.value < b.value:
            b.gradient = (b.gradient or 0) + output_tensor.gradient
        elif a.track_gradient and b.track_gradient and a.value == b.value:
            a.gradient = (a.gradient or 0) + output_tensor.gradient * 0.5
            b.gradient = (b.gradient or 0) + output_tensor.gradient * 0.5

    result.backward_function = _backward
    result.parents = [a, b]

    return result


def max_in_list(x: list):
    max_tensor = x[0]
    for tensor in x[1:]:
        max_tensor = maximum(max_tensor, tensor)

    return max_tensor


def summation(x: list):
    values = [tensor.value for tensor in x]
    track_gradient = any(tensor.track_gradient for tensor in x)
    result = Tensor(np.sum(values), track_gradient)

    def _backward(output_tensor):
        for tensor in x:
            if tensor.track_gradient:
                tensor.gradient = (tensor.gradient or 0) + output_tensor.gradient

    result.backward_function = _backward
    result.parents = x

    return result


class Tensor:
    def __init__(self, value, track_gradient=False):
        self.value = value
        self.track_gradient = track_gradient
        self.gradient = None
        self.backward_function = None
        self.parents = []

    def __repr__(self):
        return f"Tensor - value: {self.value}, gradient: {self.gradient}"

    def backward(self):
        if self.gradient is None:
            self.gradient = np.ones_like(self.value)

        if self.backward_function:
            self.backward_function(self)

        for parent in self.parents:
            if parent.backward_function:
                parent.backward()

    # overload operations
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __truediv__(self, other):
        recip = power(other, -1)
        multi = multiply(self, recip)
        return multi

    def __neg__(self):
        return negate(self)
