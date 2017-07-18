from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sign_classifier(weights, inputs):
    """
    input:
        weight: shape=[5,]
        input: shape=[5,]
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

    shuffled_index_list = np.arange(myarray.shape[0])
    total_counter = 2000
    sum_update_counter = 0
    learning_rate = 0.5
    while total_counter:
        # shuffle
        np.random.shuffle(shuffled_index_list)
        myarray = myarray[shuffled_index_list]
        total_counter -= 1

        # PLA
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
                    weight = weight + learning_rate * Y * X
        print("total update %d times" % update_counter)
        sum_update_counter += update_counter
    print("mean update %d times" % (sum_update_counter/2000.0))



if __name__ == "__main__":
    PLA()
