from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

REPEAT_TIMES = 5000


def decision_stump():
    # 1. generate training data
        # (a) Generate x by a uniform distribution in [−1,1].
        # (b) Generate y by f(x)=s~(x) + noise where s~(x)=sign(x) and the noise flips the result with 20% probability.
    # 2. looping to calculate average of E-in and E-out
        # hsθ(x)=s⋅sign(x−θ).
        # E-out:    0.5+0.3*s*(|theta| - 1)

    Ein_sum = 0.0
    Eout_sum = 0.0
    for i in range(REPEAT_TIMES):
        # step 1
        data = np.random.uniform(low=-1.0, high=1.0, size=20)
        label = (data > 0)
        for l in label:
            if np.random.uniform(0, 1) < 0.2:
                l = not l
        # step 2
        s_list = [1, -1]
        Ein_smallest = 1.0
        sorted_data = sorted(data)
        theta = sorted_data[len(sorted_data)//2]
        for s in s_list:
            Ein = 0.0
            for x, y in zip(data, label):
                logit = s * ((x - theta) > 0)
                Ein += (logit != y)
            if Ein/len(data) < Ein_smallest:
                Ein_smallest = Ein/len(data)
                b_theta = theta
                b_s = s
        Ein_sum += Ein_smallest
        # use best hypothesis to evaluate Eout
        Eout = 0.5 + 0.3 * b_s * (abs(b_theta) - 1)
        Eout_sum += Eout
    print('smallest of Ein: %f' % (Ein_sum / REPEAT_TIMES))
    print('smallest of Eout: %f' % (Eout_sum / REPEAT_TIMES))


if __name__ == "__main__":
    decision_stump()
