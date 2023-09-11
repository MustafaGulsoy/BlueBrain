import numpy as np
from math import e


def sigmoid(value):
    return 1 / (1 + e ** value)


def relu(value):
    return max(0, value)


def tanh(value):
    return 2 / (1 + e ** -2 * value) + 1


def softplus(value):
    return np.log(1 + np.exp(value))


def softmax(x):
    """Compute softmax values for each element in the input array."""
    # Ensure numerical stability by subtracting the maximum value
    # from each element before exponentiation
    x = x - np.max(x)

    # Compute the exponentiated values
    exp_values = np.exp(x)

    # Compute the softmax values by dividing the exponentiated values
    # by the sum of all exponentiated values
    softmax_values = exp_values / np.sum(exp_values)

    return softmax_values


def set_activator_func(func):
    if func == "sigmoid":
        return sigmoid
    elif func == "relu":
        return relu
    else:
        raise ValueError(f"Unknown operation: {func}")


class Layer:
    def __init__(self, input_size,output_size, activator_function: str):
        self.output_size = output_size
        self.input_size = input_size
        self.weight = [[np.random.uniform(high=1,low=-1,size=output_size)] * input_size]
        self.activator_function = set_activator_func(activator_function)


