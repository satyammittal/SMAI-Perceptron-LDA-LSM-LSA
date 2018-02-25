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

def find_category_count(array):
    arr = [0,0,0,0,0]
    min2=0
    ans = 1
    for r in array:
        e = int(r[1])
        arr[e] += 1
        if min2<arr[e]:
            min2=arr[e]
            ans=e
    return ans
    

def cosine_score(test_real, test_pred):
    total = float(len(test_real))
    same = 0.0
    for i in range(len(test_real)):
        if test_real[i]==test_pred[i]:
            same += 1.0
    return float(same/total)    
    
def cosine_model(train_arr, test_arr):
    ret = []
    test_arr2 = test_arr[:,:-1]
    test_real = test_arr[:,-1]
    for test in test_arr2:
        res  = []
        for train in train_arr:
            cos_val = 0
            val = np.dot(test, train[:-1])
            test_abs = LA.norm(test,2)
            train_abs = LA.norm(train,2)
            if test_abs !=0 and train_abs !=0:
                cos_val = 1-sc.spatial.distance.cosine(test, train[:-1])
            tup = (cos_val, train[-1])
            res.append(tup)
        res.sort(key=lambda tup: tup[0])
        data = res[-10:]
        del res
        cat = find_category_count(data)
        ret.append(cat)
    return cosine_score(test_real, ret)


def predict_cosine_model(train_arr, test_arr, loc):
    ret = []
    test = test_arr[:,:-1]
    test_real = test_arr[:,-1]
    res  = []
    i = 0
    for train in train_arr:
        cos_val = 0
        test_abs = LA.norm(test,2)
        train_abs = LA.norm(train,2)
        if test_abs !=0 and train_abs !=0:
            cos_val = 1-sc.spatial.distance.cosine(test, train[:-1])
        tup = (cos_val, train[-1], loc[i])
        res.append(tup)
        i+=1
    res.sort(key=lambda tup: tup[0])
    data = res[-10:]
    del res
    cat = find_category_count(data)
    ret.append(cat)
    print "Top related Documents: "
    for ri in xrange(len(data)-1,-1,-1):
        print len(data)-ri, " --> ", data[ri][2]
    print "Predicted:", int(ret[0])
    print "Expected:", int(test_real[0])
    return     

def cosine_cross_valid(arr):
    k = 5
    r = np.copy(arr)
    np.random.shuffle(r)
    split_arr = np.asarray(np.array_split(r, k))
    acc_mat = []
    ans_arr = []
    for j in range(k):
        test_arr = split_arr[j]
        train_arr = merge_arrays(split_arr[:j])
        t = merge_arrays(split_arr[j+1:])
        train_arr = merge_arrays([train_arr,t])
        accuracy = cosine_model(train_arr, test_arr)
        ans_arr.append(accuracy)
    larr = np.array(ans_arr)
    print (larr)
    print ("{0} +/- {1}").format(larr.mean(), 2*larr.std())
    acc_mat.append(larr.mean())
    return acc_mat

if __name__ == "__main__":
    vectors_load = np.load('tfidf_reduced.npy')
    mt = sc.sparse.csc_matrix(vectors_load.all())
    tfidf_mat = mt.todense()
    model = cosine_cross_valid(np.asarray(tfidf_mat))
    print model
    pass