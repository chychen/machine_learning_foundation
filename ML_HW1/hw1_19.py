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


def pocket_PLA():
    """
    implementation of Pocket Percepture Learning Algorithm
    """
    train_data = np.loadtxt('hw1_18_train.txt')
    test_data = np.loadtxt('hw1_18_test.txt')
    print("input shape:", train_data.shape)
    print("input dtype:", train_data.dtype)

    shuffled_index_list = np.arange(train_data.shape[0])
    total_counter = 2000
    sum_error_rate = 0.0
    while total_counter:
        # shuffle
        np.random.shuffle(shuffled_index_list)
        train_data = train_data[shuffled_index_list]
        total_counter -= 1

        # pocket PLA
        update_counter = 0
        weight = np.zeros(shape=[5], dtype=float)
        res_weight = weight
        # training
        success_predict_counter = 0
        while update_counter < 50:
            for _, v in enumerate(train_data):
                # X[0] as bias
                X = np.concatenate([[1], v[0:4]])
                Y = v[-1]
                if sign_classifier(weight, X) != int(Y):
                    weight_candidate = weight + Y * X
                    weight = weight_candidate
                    update_counter += 1
                    # verify weight_candidate is better or not
                    candidate_success_predict_counter = 0
                    for _, v2 in enumerate(train_data):
                        X2 = np.concatenate([[1], v2[0:4]])
                        Y2 = v2[-1]
                        if sign_classifier(weight_candidate, X2) == int(Y2):
                            candidate_success_predict_counter += 1
                    if candidate_success_predict_counter > success_predict_counter:
                        success_predict_counter = candidate_success_predict_counter
                        res_weight = weight_candidate
                        # print("find better!!")
                if update_counter >= 50:
                    break
        # testing
        error_counter = 0
        for _, v in enumerate(test_data):
            X = np.concatenate([[1], v[0:4]])
            Y = v[-1]
            if sign_classifier(weight, X) != int(Y):
                error_counter += 1
        error_rate = error_counter / test_data.shape[0]
        print("error rate %f" % error_rate)
        print("----------------------------------")
        sum_error_rate += error_rate
        # input()
    print("mean error rate %f" % (sum_error_rate / 2000.0))


if __name__ == "__main__":
    pocket_PLA()
