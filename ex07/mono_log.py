#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def check_matrix(matrix):
    if isinstance(matrix, np.ndarray) and matrix.size != 0 and len(matrix.shape) == 2:
        return True
    exit("Error matrix")


def check_vector(vector):
    if (
        isinstance(vector, np.ndarray)
        and vector.size != 0
        and len(vector.shape) == 2
        and vector.shape[1] == 1
    ):
        return True
    exit("Error vector")


def data_spliter(x, y, proportion):
    if (
        check_matrix(x)
        and check_vector(y)
        and x.shape[0] == y.shape[0]
        and isinstance(proportion, float)
        and proportion <= 1
    ):
        df = np.hstack((x, y))
        np.random.shuffle(df)
        x_train, x_test = np.split(df[:, :-1], [int(proportion * x.shape[0])])
        y_train, y_test = np.split(df[:, [-1]], [int(proportion * y.shape[0])])
        return (x_train, x_test, y_train, y_test)
    return


class MyLogisticRegression:
    """
    Description:
    My personal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=1e-4, max_iter=10000):
        if (
            check_vector(theta)
            and isinstance(max_iter, int)
            and isinstance(alpha, float)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
        else:
            return

    def sigmoid_(self, x):
        if isinstance(x, np.ndarray) and x.size != 0 and x.shape == ():
            return [[1 / (1 + np.exp(-x))]]
        if check_vector(x):
            return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        if (
            check_matrix(x)
            and check_vector(self.theta)
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return self.sigmoid_(x @ self.theta)

    def loss_(self, x, y):
        eps = 1e-15
        y_hat = self.predict_(x)
        if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
            one = np.ones(y.shape)
            return (
                -(y.T @ np.log(y_hat + eps) + (one - y).T @ np.log(one - y_hat + eps))
                / y.shape[0]
            )[0][0]

    def fit_(self, x, y):
        if (
            check_matrix(x)
            and check_vector(y)
            and check_vector(self.theta)
            and x.shape[0] == y.shape[0]
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = x.T @ (self.sigmoid_(x @ self.theta) - y) / x.shape[0]
                self.theta -= self.alpha * gradient
            return self.theta


def parse():
    parser = argparse.ArgumentParser(
        description="Discriminate between citizens who come from your favorite planet and everybody else",
    )
    parser.add_argument(
        "-zipcode",
        help="x being 0, 1, 2 or 3",
        metavar="x",
        required=True,
        type=int,
    )
    parser.parse_args(["-zipcode", "-1"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    zipcode = args._get_kwargs()[0][1]
    if zipcode < 0 or zipcode > 3:
        exit("x must be 0, 1, 2 or 3")

    try:
        if os.stat("../resources/solar_system_census.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/solar_system_census.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
            planets = np.loadtxt(
                "../resources/solar_system_census_planets.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    df = np.hstack((data[:, 1:], planets[:, 1:]))
    (x, x_test, y, y_test) = data_spliter(df[:, :-1], df[:, [-1]], 0.7)

    mlr = MyLogisticRegression(np.ones(x.shape[1] + 1).reshape(-1, 1))
    mlr.fit_(x, np.where(y == zipcode, 1, 0))

    prediction = np.where(mlr.predict_(x_test) > 0.5, True, False)
    pos = y_test == zipcode
    unique, counts = np.unique(pos == prediction, return_counts=True)
    print(counts[1] / y_test.shape[0])

    figure, axis = plt.subplots(1, 3)
    y_label = ["height", "weight", "bone_density"]
    for i in range(3):
        axis[i].set_xlabel("x")
        axis[i].set_ylabel(y_label[i])
        axis[i].scatter(x[:, i], np.where(y == zipcode, 1, 0), label="dataset")
        axis[i].scatter(x_test[:, i], prediction, marker=".", label="prediction")
        axis[i].legend()
    plt.show()
