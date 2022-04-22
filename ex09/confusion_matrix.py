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


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
        iter = np.array(labels) if labels else np.unique(np.concatenate((y, y_hat)))
        ret = np.zeros((iter.shape[0], iter.shape[0]))

        for i, label in enumerate(iter):
            for j, pos_label in enumerate(iter):
                unique, counts = np.unique(
                    (y == label) & (y_hat == pos_label), return_counts=True
                )
                res = dict(zip(unique, counts))
                ret[i, j] = res.get(True) if res.get(True) else 0

        if df_option:
            ret = np.vstack([iter, ret])
            ret = np.hstack([np.reshape(np.append("", iter), (-1, 1)), ret])
            size = len(max(iter, key=len)) + 1
            res = ""
            for i in range(ret.shape[0]):
                for j in range(ret.shape[1]):
                    res += (
                        str(ret[j, i]).ljust(size)
                        if j == 0
                        else str(ret[j, i]).rjust(size)
                    )
                res += "\n"
            ret = res
        return ret


from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    y_hat = np.array(["norminet", "dog", "norminet", "norminet", "dog", "bird"])
    y = np.array(["dog", "dog", "norminet", "norminet", "dog", "norminet"])
    print("Example 1:")
    print(confusion_matrix_(y, y_hat))
    ## Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    print(confusion_matrix_(y, y_hat) == confusion_matrix(y, y_hat))

    print("Example 2:")
    print(confusion_matrix_(y, y_hat, labels=["dog", "norminet"]))
    ## Output:
    # array([[2 1]
    # [0 2]])
    print(
        confusion_matrix_(y, y_hat, labels=["dog", "norminet"])
        == confusion_matrix(y, y_hat, labels=["dog", "norminet"])
    )

    print("Example 3:")
    print(confusion_matrix_(y, y_hat, df_option=True))
    # Output:
    # bird dog norminet
    # bird 0 0 0
    # dog 0 2 1
    # norminet 1 0 2

    print("Example 4:")
    print(confusion_matrix_(y, y_hat, labels=["bird", "dog"], df_option=True))
    # Output:
    # bird dog
    # bird 0 0
    # dog 0 2
