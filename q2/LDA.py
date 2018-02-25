import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import sys

def find_threshhold(mat, class1, class2):
    t1 = -sys.maxint
    t2 = sys.maxint
    mat = np.asarray(mat)[:,0]
    for r1 in class1:
        t1=max(t1,-(np.dot(mat,r1)))
    for r2 in class2:
        t2=min(t2,-1*(np.dot(mat,r2)))
    return (t1+t2)/2


def fisher_lda(x, y):
    class1 = []
    class2 = []
    for a in zip(x,y):
        if a[1]==1:
            class1.append(a[0])
        else:
            class2.append(a[0])
    mat1 = np.asmatrix(class1)
    mat2 = np.asmatrix(class2)
    u1 = mat1.mean(0)
    u2 = mat2.mean(0)
    reg1 = mat1 - u1
    reg2 = mat2 - u2
    sb2 = np.matmul(reg1.transpose(), reg1) + np.matmul(reg2.transpose(), reg2)
    u = (u1-u2).transpose()
    ans = np.matmul(np.linalg.inv(sb2),u)
    ans /= np.linalg.norm(ans)
    b = find_threshhold(np.asarray(ans), class1, class2)
    ans =  np.vstack([np.asarray(ans), b])
    return ans