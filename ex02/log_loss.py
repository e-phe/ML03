#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from log_pred import logistic_predict_
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


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Return:
    The logistic loss value as a float.
    None otherwise.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
        return (
            -np.sum(y.T @ np.log(y_hat + eps) + (1 - y).T @ np.log(1 - y_hat + eps))
            / y.shape[0]
        )


if __name__ == "__main__":
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(log_loss_(y1, y_hat1))
    # Output: 0.01814992791780973

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(log_loss_(y2, y_hat2))
    # Output: 2.4825011602474483

    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(log_loss_(y3, y_hat3))
    # Output: 2.9938533108607053
