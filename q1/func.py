import numpy as np
import sys
import random
random.seed(9001)
eps = sys.float_info.epsilon

def sign(num):
    if num<=0:
        return -1
    else:
        return 1

def get_rid_of_nulls(value):
    if value=='?':
        return None
    else:
        return value

def merge_arrays(arr):
    out = np.array([])
    #print out.shape
    for r in arr:
        if r.shape[0]==0:
            pass
        elif out.shape[0]==0:
            out = r
        else:
            out = np.concatenate((out, r), axis=0)
    #print out.shape
    return out

def score(test_real, test_pred):
    total = float(len(test_real))
    same = 0.0
    for i in range(len(test_real)):
        if test_real[i]==test_pred[i]:
            same += 1.0
    return float(same/total)