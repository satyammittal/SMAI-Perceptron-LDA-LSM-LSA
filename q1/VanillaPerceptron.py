import numpy as np
import sys
import random
from func import *
random.seed(9001)
eps = sys.float_info.epsilon

def VanillaPerceptron(array, epoch=1):
    n = array.shape[1]
    w = np.zeros(n-1)
    b = 0
    counter=0
    for t in range(epoch):
        for x in array:
            counter+=1
            y_pred = np.dot(x[:-1],w) + b
            if (y_pred * x[-1]) <= 0:
                w += x[:-1] * x[-1]
                b += x[-1]
            #print (counter, w, b)
    return (np.copy(w),b)
    
    
def predictVanilla(model, data):
    val = np.dot(model[0],data)+model[1]
    return sign(val)

def vanilla_perceptron_model(train, test, epoch=1):
    model = VanillaPerceptron(train, epoch)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVanilla(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    return score(test_real, test_out)