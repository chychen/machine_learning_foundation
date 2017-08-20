from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

REPEAT_TIMES = 5000


def decision_stump():
    
    train_data = np.loadtxt('hw2_19_train.txt')
    test_data = np.loadtxt('hw2_19_train.txt')
    print("input shape:", train_data.shape)
    print("input dtype:", train_data.dtype)
    train_input = train_data[:,:-1]
    train_label = train_data[:,-1]
    test_input = test_data[:,:-1]
    test_label = test_data[:,-1]
    
    # TODO nothing but use one-dimention method to calculate each dim, and find the best hypothesis

if __name__ == "__main__":
    decision_stump()
