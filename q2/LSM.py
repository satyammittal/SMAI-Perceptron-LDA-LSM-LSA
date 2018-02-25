import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import sys

def least_squared_approach(x, y):
    r = x.shape[0]
    z = np.zeros((r,1), dtype=np.int64)
    z += 1
    m = np.append(x, z, axis=1)
    A = np.asmatrix(m)
    At = A.transpose()
    inver = inv(np.matmul(At , A) )
    Y = np.asmatrix(y).transpose()
    return np.matmul(np.matmul(inver, At), Y)