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


def check_matrix(matrix):
    if isinstance(matrix, np.ndarray) and matrix.size != 0 and len(matrix.shape) == 2:
        return True
    exit("Error matrix")


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * n.
    theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
    y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
    with expected and compatible shapes.
    None: otherwise.
    Raises:
    This function should not raise any Exception.
    """
    if check_matrix(x) and check_vector(theta) and x.shape[1] + 1 == theta.shape[0]:
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return sigmoid_(x @ theta)


if __name__ == "__main__":
    x = np.array([[4]])
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))
    # Output: array([[0.98201379]])

    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))
    # Output: array([[0.98201379],[0.99624161],[0.97340301],[0.99875204],[0.90720705]])

    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
    # Output: array([[0.03916572],[0.00045262],[0.2890505 ]])
