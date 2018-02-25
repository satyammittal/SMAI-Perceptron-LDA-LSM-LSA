import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import linalg as LA
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import scipy as sc
from dataset import retreive_file
from cosine_similarity import predict_cosine_model

if __name__ == "__main__":
    train, test, maptoloc = retreive_file(sys.argv[1], sys.argv[2], sys.argv[3])
    predict_cosine_model(train, test, maptoloc)