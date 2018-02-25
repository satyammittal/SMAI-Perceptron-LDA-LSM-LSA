import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import linalg as LA
from nltk.stem import PorterStemmer
#reload(sys)
#sys.setdefaultencoding("utf-8")
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.feature_extraction.text import TfidfTransformer

min_epoch = 5
diff_epoch = 5
max_epoch = 20

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
    return np.append(np.copy(w),b)

def makeitsingular(number, k):
    if number == k:
        return 1
    else:
        return -1

def multiclassPerceptron(array, epoch, numclasses):
    label = array[:,-1].copy()
    w = []
    for k in range(numclasses):
        clas = k
        arr = np.array([makeitsingular(t, clas) for t in label])
        array[:, -1] = arr
        w.append(VanillaPerceptron(array, epoch))
    return w
    
    
def predictVanilla(model, data):
    y = data.copy()
    y = np.append(y,1)
    minh = -10000000000007.0
    pred = 1
    counter = 0
    for m in model:
        val = float(np.dot(m,y))
        if minh < val:
            minh = val
            pred = counter
        counter += 1
    return pred

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

def vanilla_perceptron_model(train, test, epoch=1):
    model = multiclassPerceptron(train.copy(), epoch, 5)
    train_real = train[:,-1]
    train_in = np.delete(train, -1, axis=1)
    train_out = []
    for val in train_in:
        pred = predictVanilla(model, val)
        train_out.append(pred)
    print "Training Accuracy-> ", score(train_real, train_out)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVanilla(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    return score(test_real, test_out)

def sign(num):
    if num<=0:
        return -1
    else:
        return 1
    
def perceptron_cross_valid(arr, string):
    k = 5
    r = np.copy(arr)
    np.random.shuffle(r)
    split_arr = np.asarray(np.array_split(r, k))
    acc_mat = []
    for i in xrange(min_epoch, max_epoch+1, diff_epoch):
        ans_arr = []
        for j in range(k):
            test_arr = split_arr[j]
            train_arr = merge_arrays(split_arr[:j])
            t = merge_arrays(split_arr[j+1:])
            train_arr = merge_arrays([train_arr,t])
            accuracy = vanilla_perceptron_model(train_arr, test_arr, i)
            ans_arr.append(accuracy)
        larr = np.array(ans_arr)
        print ("Epoch {0} -> {1} +/- {2}").format(i, larr.mean(), 2*larr.std())
        acc_mat.append(larr.mean())
    return acc_mat

if __name__ == "__main__":
    vectors_load = np.load('tfidf_reduced.npy')
    mt = sc.sparse.csc_matrix(vectors_load)
    tfidf_mat = mt.todense()
    model = perceptron_cross_valid(np.asarray(tfidf_mat), "vanilla")
    print model
    pass