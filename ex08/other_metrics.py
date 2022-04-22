#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_vector(vector):
    if (
        isinstance(vector, np.ndarray)
        and vector.size != 0
        and (
            (len(vector.shape) == 2 and vector.shape[1] == 1)
            or (len(vector.shape) == 1)
        )
    ):
        return True
    exit("Error vector")


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    Return:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
        unique, counts = np.unique(y == y_hat, return_counts=True)
        return counts[1] / y.shape[0]


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if (
        check_vector(y)
        and check_vector(y_hat)
        and y.shape == y_hat.shape
        and (isinstance(pos_label, str) or isinstance(pos_label, int))
    ):
        pos = y == pos_label
        pos_hat = y_hat == pos_label
        unique, tp = np.unique(pos & pos_hat, return_counts=True)
        unique, all_pos = np.unique(pos_hat, return_counts=True)
        return tp[1] / all_pos[1]


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if (
        check_vector(y)
        and check_vector(y_hat)
        and y.shape == y_hat.shape
        and (isinstance(pos_label, str) or isinstance(pos_label, int))
    ):
        pos = y == pos_label
        pos_hat = y_hat == pos_label
        unique, tp = np.unique(pos & pos_hat, return_counts=True)
        fn = sum(y_hat[i] != pos_label and y[i] != y_hat[i] for i in range(y.shape[0]))
        return tp[1] / (tp[1] + int(fn))


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if (
        check_vector(y)
        and check_vector(y_hat)
        and y.shape == y_hat.shape
        and (isinstance(pos_label, str) or isinstance(pos_label, int))
    ):
        return (
            2
            * precision_score_(y, y_hat, pos_label)
            * recall_score_(y, y_hat, pos_label)
        ) / (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    print("Example 1:")
    y_hat = np.array([[1], [1], [0], [1], [0], [0], [1], [1]])
    y = np.array([[1], [0], [0], [1], [0], [1], [0], [0]])

    print("\nAccuracy")
    print(accuracy_score_(y, y_hat))
    ## Output: 0.5
    print(accuracy_score_(y, y_hat) == accuracy_score(y, y_hat))

    print("\nPrecision")
    print(precision_score_(y, y_hat))
    ## Output: 0.4
    print(precision_score_(y, y_hat) == precision_score(y, y_hat))

    print("\nRecall")
    print(recall_score_(y, y_hat))
    ## Output: 0.6666666666666666
    print(recall_score_(y, y_hat) == recall_score(y, y_hat))

    print("\nF1-score")
    print(f1_score_(y, y_hat))
    ## Output: 0.5
    print(f1_score_(y, y_hat) == f1_score(y, y_hat))

    print("\nExample 2:")
    y_hat = np.array(
        ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
    )
    y = np.array(
        ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
    )

    print("\nAccuracy")
    print(accuracy_score_(y, y_hat))
    ## Output: 0.625
    print(accuracy_score_(y, y_hat) == accuracy_score(y, y_hat))

    print("\nPrecision")
    print(precision_score_(y, y_hat, pos_label="dog"))
    ## Output: 0.6
    print(
        precision_score_(y, y_hat, pos_label="dog")
        == precision_score(y, y_hat, pos_label="dog")
    )

    print("\nRecall")
    print(recall_score_(y, y_hat, pos_label="dog"))
    ## Output: 0.75
    print(
        recall_score_(y, y_hat, pos_label="dog")
        == recall_score(y, y_hat, pos_label="dog")
    )

    print("\nF1-score")
    print(f1_score_(y, y_hat, pos_label="dog"))
    ## Output: 0.6666666666666665
    print(f1_score_(y, y_hat, pos_label="dog") == f1_score(y, y_hat, pos_label="dog"))

    print("\nExample 3:")
    y_hat = np.array(
        ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
    )
    y = np.array(
        ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
    )

    print("\nAccuracy")
    print(accuracy_score_(y, y_hat))
    ## Output: 0.625
    print(accuracy_score_(y, y_hat) == accuracy_score(y, y_hat))

    print("\nPrecision")
    print(precision_score_(y, y_hat, pos_label="norminet"))
    ## Output: 0.6666666666666666
    print(
        precision_score_(y, y_hat, pos_label="norminet")
        == precision_score(y, y_hat, pos_label="norminet")
    )

    print("\nRecall")
    print(recall_score_(y, y_hat, pos_label="norminet"))
    ## Output: 0.5
    print(
        recall_score_(y, y_hat, pos_label="norminet")
        == recall_score(y, y_hat, pos_label="norminet")
    )

    print("\nF1-score")
    print(f1_score_(y, y_hat, pos_label="norminet"))
    ## Output: 0.5714285714285715
    print(
        f1_score_(y, y_hat, pos_label="norminet")
        == f1_score(y, y_hat, pos_label="norminet")
    )
