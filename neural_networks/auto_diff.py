import numpy as np
from numpy.typing import NDArray


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


def add(a: Tensor, b: Tensor) -> Tensor:
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(np.add(a.value, b.value), track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient
        if b.track_gradient:
            b.gradient = (b.gradient or 0) + output_tensor.gradient

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def subtract(a: Tensor, b: Tensor) -> Tensor:
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(np.subtract(a.value, b.value), track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient
        if b.track_gradient:
            b.gradient = (b.gradient or 0) - output_tensor.gradient

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def multiply(a: Tensor, b: Tensor) -> Tensor:
    track_gradient = any([a.track_gradient, b.track_gradient])
    result = Tensor(np.multiply(a.value, b.value), track_gradient)

    def _backward(output_tensor):
        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient * b.value
        if b.track_gradient:
            b.gradient = (b.gradient or 0) + output_tensor.gradient * a.value

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def power(base: Tensor, exponent: float) -> Tensor:
    track_gradient = base.track_gradient
    base_value = (
        base.value.astype(float)
        if isinstance(base.value, np.ndarray)
        else float(base.value)
    )

    result = Tensor(np.power(base_value, exponent), track_gradient)

    def _backward(output_tensor):
        base_value = (
            base.value.astype(float)
            if isinstance(base.value, np.ndarray)
            else float(base.value)
        )
        if base.track_gradient:
            base.gradient = (base.gradient or 0) + output_tensor.gradient * exponent * (
                np.power(base_value, (exponent - 1))
            )

    result.backward_function = _backward
    result.parents = [base]
    return result


def exp(exponent: Tensor) -> Tensor:
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


def negate(x: Tensor) -> Tensor:
    result = Tensor(-x.value, x.track_gradient)

    def _backward(output_tensor):
        if x.track_gradient:
            x.gradient = (x.gradient or 0) - output_tensor.gradient

    result.backward_function = _backward
    result.parents = [x]

    return result


def maximum(a: Tensor, b: Tensor) -> Tensor:
    track_gradient = any([a.track_gradient, b.track_gradient])
    result_value = np.maximum(a.value, b.value)
    result = Tensor(result_value, track_gradient)

    def _backward(output_tensor):
        mask_a = a.value > b.value
        mask_b = b.value > a.value
        # Handle the case where a and b are equal by dividing the gradient
        mask_equal = a.value == b.value

        if a.track_gradient:
            a.gradient = (a.gradient or 0) + output_tensor.gradient * (
                mask_a + 0.5 * mask_equal
            )
        if b.track_gradient:
            b.gradient = (b.gradient or 0) + output_tensor.gradient * (
                mask_b + 0.5 * mask_equal
            )

    result.backward_function = _backward
    result.parents = [a, b]
    return result


def reduce_max(tensor: Tensor) -> Tensor:
    max_value = tensor.value[0]
    max_index = 0

    for i in range(1, len(tensor.value)):
        if tensor.value[i] > max_value:
            max_value = tensor.value[i]
            max_index = i

    result = Tensor(max_value, track_gradient=tensor.track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:

            tensor.gradient = (
                tensor.gradient
                if tensor.gradient is not None
                else np.zeros_like(tensor.value)
            )
            tensor.gradient[max_index] += output_tensor.gradient

    result.backward_function = _backward
    result.parents = [tensor]

    return result


def summation(tensor: Tensor) -> Tensor:
    values = tensor.value
    track_gradient = tensor.track_gradient
    result = Tensor(np.sum(values), track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:
            tensor.gradient = (tensor.gradient or 0) + output_tensor.gradient

    result.backward_function = _backward
    result.parents = [tensor]

    return result


def mean(x: Tensor) -> Tensor:
    total_sum = summation(x)
    count = Tensor(len(x.value))
    return total_sum / count


def sqrt(tensor: Tensor) -> Tensor:
    return power(tensor, 0.5)


def absolute(tensor: Tensor) -> Tensor:
    track_gradient = tensor.track_gradient
    result = Tensor(np.abs(tensor.value), track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:
            tensor.gradient = (tensor.gradient or 0) + output_tensor.gradient * np.sign(
                tensor.value
            )

    result.backward_function = _backward
    result.parents = [tensor]

    return result


def log(tensor: Tensor) -> Tensor:
    track_gradient = tensor.track_gradient
    result = Tensor(np.log(tensor.value), track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:
            tensor.gradient = (
                tensor.gradient or 0
            ) + output_tensor.gradient / tensor.value

    result.backward_function = _backward
    result.parents = [tensor]

    return result


def relu(tensor: Tensor) -> Tensor:
    result_value = np.maximum(tensor.value, 0)
    result = Tensor(result_value, tensor.track_gradient)

    def _backward(output_tensor):
        # Gradient is 1 where tensor.value > 0, else 0
        if tensor.track_gradient:
            mask = (
                (tensor.value > 0).astype(float)
                if isinstance(tensor.value, np.ndarray)
                else float(tensor.value > 0)
            )
            tensor.gradient = (tensor.gradient or 0) + output_tensor.gradient * mask

    result.backward_function = _backward
    result.parents = [tensor]
    return result


def softmax(tensor: Tensor) -> Tensor:
    max_value = np.max(tensor.value)
    exp_values = np.exp(tensor.value - max_value)
    softmax_values = exp_values / np.sum(exp_values)
    result = Tensor(softmax_values, track_gradient=tensor.track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:
            if tensor.gradient is None:
                tensor.gradient = np.zeros_like(tensor.value)

            # Compute the Jacobian matrix for softmax
            s = softmax_values.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            tensor.gradient += np.dot(jacobian, output_tensor.gradient)

    result.backward_function = _backward
    result.parents = [tensor]
    return result


def tanh(tensor: Tensor) -> Tensor:
    tanh_value = np.tanh(tensor.value)
    result = Tensor(tanh_value, track_gradient=tensor.track_gradient)

    def _backward(output_tensor):
        if tensor.track_gradient:
            if tensor.gradient is None:
                tensor.gradient = np.zeros_like(tensor.value)
            tensor.gradient += output_tensor.gradient * (1 - tanh_value**2)

    result.backward_function = _backward
    result.parents = [tensor]
    return result


def binary_cross_entropy(truth, prediction):
    # Clip prediction values for numerical stability
    prediction_value = np.clip(prediction.value, 1e-15, 1 - 1e-15)
    bce_value = -np.mean(
        truth.value * np.log(prediction_value)
        + (1 - truth.value) * np.log(1 - prediction_value)
    )

    result = Tensor(
        bce_value, track_gradient=truth.track_gradient or prediction.track_gradient
    )

    def _backward(output_tensor):
        if truth.track_gradient or prediction.track_gradient:
            if truth.track_gradient and truth.gradient is None:
                truth.gradient = np.zeros_like(truth.value, dtype=np.float64)
            if prediction.track_gradient and prediction.gradient is None:
                prediction.gradient = np.zeros_like(prediction.value, dtype=np.float64)

            # Gradient for prediction: - (truth / prediction - (1 - truth) / (1 - prediction)) / N
            N = truth.value.size
            grad_prediction = (
                -(
                    truth.value / prediction_value
                    - (1 - truth.value) / (1 - prediction_value)
                )
                / N
            )

            if prediction.track_gradient:
                prediction.gradient += grad_prediction * output_tensor.gradient

            if truth.track_gradient:
                grad_truth = (
                    -np.log(prediction_value + 1e-15) / N
                    + np.log(1 - prediction_value + 1e-15) / N
                )
                truth.gradient += grad_truth * output_tensor.gradient

    result.backward_function = _backward
    result.parents = [truth, prediction]
    return result
