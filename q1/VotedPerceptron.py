import numpy as np
import sys
import random
from func import *
random.seed(9001)
eps = sys.float_info.epsilon

def VotedPerceptron(array, epoch=1):
    n = array.shape[1]
    w = np.zeros(n-1)
    b = 0
    c = 0
    ans = []
    for t in range(epoch):
        for x in array:
            y_pred = np.dot(x[:-1],w) + b
            if y_pred * x[-1] > 0:
                c = c + 1
            elif c != 0:
                ret = (np.copy(w), b, c)
                ans.append(ret)
                w += x[:-1] * x[-1]
                b += x[-1]
                c = 1
            else:
                c += 1
                w += x[:-1] * x[-1]
                b += x[-1]
        ans.append((np.copy(w),b,c))
    return ans

def predictVoted(model, data):
    val = 0
    for m in model:
        val += m[2]*sign(np.dot(m[0],data)+m[1])
    return sign(val)

def voted_perceptron_model(train, test, epoch=1):
    model = VotedPerceptron(train, epoch)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVoted(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    return score(test_real, test_out)