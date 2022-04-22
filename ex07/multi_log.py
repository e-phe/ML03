#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

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


def zscore(x):
    if check_vector(x):
        mean = sum(x) / x.shape[0]
        std = np.sqrt(np.square(x - mean).sum() / x.shape[0])
        x = (x - mean) / std
        return x


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

    def __init__(self, theta, alpha=1e-6, max_iter=10000):
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


if __name__ == "__main__":
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
    for i in range(df.shape[1] - 1):
        df[:, [i]] = zscore(df[:, [i]])
    (x, x_test, y, y_test) = data_spliter(df[:, :-1], df[:, [-1]], 0.7)

    thetas = []
    for zipcode in range(4):
        mlr = MyLogisticRegression(np.ones(x.shape[1] + 1).reshape(-1, 1))
        y_ones = np.where(y == zipcode, 1, 0)
        thetas.append(mlr.fit_(x, y_ones))

    planets = [
        "The flying cities of Venus",
        "United Nations of Earth",
        "Mars Republic",
        "The Asteroids' Belt colonies ",
    ]
    x_pred = np.insert(x, 0, values=1.0, axis=1).astype(float)
    origins = [
        max(
            (i @ np.array(thetas[zipcode]), planet)
            for zipcode, planet in enumerate(planets)
        )[1]
        for i in x_pred
    ]
    print(origins)

    x_pred_test = np.insert(x_test, 0, values=1.0, axis=1).astype(float)
    y_hat = np.array(
        [
            max((i @ np.array(thetas[zipcode]), zipcode) for zipcode in range(4))[1]
            for i in x_pred_test
        ]
    ).reshape(-1, 1)
    unique, counts = np.unique(y_test == y_hat, return_counts=True)
    print(counts[1] / y_test.shape[0])

    figure, axis = plt.subplots(1, 3)
    y_label = ["height", "weight", "bone_density"]
    prediction = np.where(mlr.predict_(x_test) > 0.5, 1, 0)
    for i in range(3):
        axis[i].set_xlabel("x")
        axis[i].set_ylabel(y_label[i])
        axis[i].scatter(x[:, i], y_ones, label="y")
        axis[i].scatter(x_test[:, i], prediction, marker=".", label="prediction")
        axis[i].legend()
    plt.show()
