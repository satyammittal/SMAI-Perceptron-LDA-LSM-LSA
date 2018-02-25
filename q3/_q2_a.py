import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import linalg as LA
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import scipy as sc
from dataset import retreive_doc
from Perceptron import vanilla_perceptron_model
from sklearn.feature_extraction.text import TfidfTransformer

def give_reduced_matrix(dataset):
    AAt = np.matmul(dataset[:,:-1].transpose(),dataset[:,:-1])
    w, v = sc.linalg.eig(AAt)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    U = v[:,:-1*5]
    R = np.matmul(dataset[:,:-1], U)
    reduced = np.append(R, dataset[:,-1], axis=1)
    return reduced



if __name__ == "__main__":
    train = retreive_doc(sys.argv[1])
    train = give_reduced_matrix(train)
    test = retreive_doc(sys.argv[2])
    test = give_reduced_matrix(test)
    vanilla_perceptron_model(np.asarray(train), np.asarray(test), 5)