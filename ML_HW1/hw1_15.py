from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sign_classifier(weights, inputs):
    """
    input:
        weight: shape=[4,]
        input: shape=[4,]
    """
    result = np.dot(weights, inputs)
    if result > 0:
        return 1
    else:
        return -1


def PLA():
    """
    implementation of Percepture Learning Algorithm
    """
    myarray = np.loadtxt('hw1_15_train.txt')
    print("input shape:", myarray.shape)
    print("input dtype:", myarray.dtype)

    update_counter = 0
    weight = np.zeros(shape=[5], dtype=float)
    if_stop_update = False
    while not if_stop_update:
        if_stop_update = True
        for _, v in enumerate(myarray):
            # X[0] as bias
            X = np.concatenate([[1], v[0:4]])
            Y = v[-1]
            if sign_classifier(weight, X) != int(Y):
                if_stop_update = False
                update_counter += 1
                weight = weight + Y * X

    print("total update %d times" % update_counter)


if __name__ == "__main__":
    PLA()
