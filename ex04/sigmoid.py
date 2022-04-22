#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_vector(vector):
    if (
        isinstance(vector, np.ndarray)
        and vector.size != 0
        and len(vector.shape) == 2
        and vector.shape[1] == 1
    ):
        return True
    exit("Error vector")


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.array, a vector
    Return:
    The sigmoid value as a numpy.array.
    None otherwise.
    Raises:
    This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) and x.size != 0 and x.shape == ():
        return [[1 / (1 + np.exp(-x))]]
    if check_vector(x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.array(-4)
    print(sigmoid_(x))
    # Output: array([[0.01798620996209156]])

    x = np.array(2)
    print(sigmoid_(x))
    # Output: array([[0.8807970779778823]])

    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
    # Output: array([[0.01798620996209156], [0.8807970779778823], [0.5]])
