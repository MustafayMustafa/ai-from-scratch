import math


def sigmoid(value):
    # two formulations to handle numerical stabillity for both large positive and negative values
    if value > 0:
        return 1 / (1 + math.pow(math.e, -value))
    else:
        return math.pow(math.e, value) / (1 + math.pow(math.e, value))


def relu(value):
    raise NotImplementedError


def softmax(value):
    raise NotImplementedError
