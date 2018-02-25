import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import linalg as LA
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import scipy as sc
from scipy import linalg
from sklearn.feature_extraction.text import TfidfTransformer
from Perceptron import perceptron_cross_valid
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def singularvector(minf,diff,maxf):
    lenf = []
    for w in xrange(minf, maxf, diff):
        lenf.append(dataset.shape[1]-1-w)
    return lenf

def check_on_various_features(dataset, minf, diff, maxf, v):
    ans = []
    for w in xrange(minf, maxf, diff):
        U = v[:,:-1*w]
        R = np.matmul(dataset[:,:-1], U)
        reduced = np.append(R, dataset[:,-1], axis=1)
        accuracy = perceptron_cross_valid(np.asarray(reduced), "vanilla")
        print accuracy
        ans.append(accuracy)
    return ans

if __name__ == "__main__":
    vectors_load = np.load('tfidf_reduced.npy')
    mt = sc.sparse.csc_matrix(vectors_load.all())
    dataset = mt.todense()
    AAt = np.matmul(dataset[:,:-1].transpose(),dataset[:,:-1])
    w, v = linalg.eig(AAt)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    res = check_on_various_features(dataset,3550,1,3588,v.astype('float'))
    len2 = singularvector(3550,1,3588)
    fig = plt.figure()
    plt.xlabel('Number of Singular Vectors')
    plt.ylabel('Accuracy')
    plt.plot(len2,res, '.r-', label='Perceptron')
    plt.legend(loc='best')
    fig.set_size_inches(12, 8)
    plt.title('SVD')
    plt.savefig("q3/q3_part1.jpg")
    plt.show()