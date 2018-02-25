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

def get_bow(loc):
    ans = {}
    dirs = os.listdir(loc)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    df = pd.DataFrame(columns=['label_set'])
    counter=0
    for ind in dirs:
        files = os.listdir(loc+'/'+ind)
        for f in files:
            df.loc[counter]=0
            df.loc[counter, 'DB:label'] = int(ind)
            floc = loc+'/'+ind+'/'+f
            #print floc
            df.loc[counter, 'DB:loc'] = str(floc)
            with open(floc, 'r') as content_file:
                content = content_file.read()
                content=str(content).decode('UTF-8', 'ignore').lower()
                word_tokens = word_tokenize(content)
                filtered_sentence = [ps.stem(w) for w in word_tokens if not w in stop_words and not ps.stem(w) in stop_words and w.isalnum()]
                rt = {}
                for word in filtered_sentence:
                    if word not in rt:
                        rt[word]=0
                    rt[word] += 1

                for word in rt:
                    if word not in df.columns:
                        df[word]=0
                    df.loc[counter, word] += rt[word]
            counter += 1
    return df

def add_labels(loc, df):
    ans = {}
    dirs = os.listdir(loc)
    label = []
    for ind in dirs:
        files = os.listdir(loc+'/'+ind)
        for f in files:
            label.append(ind)
    label = np.array(label)
    df['label'] = label
    return

def drop_words(dr):
    count=0
    arr = []
    for r in dr.columns:
        if dr[dr[r]!=0].shape[0] < 10 and r not in ['doc_id','DB:label','label_set','DB:loc']:
            arr.append(r)
        count+=1
    dr = dr.drop(arr, axis=1)
    return dr

def retreive_doc(location):
    final_bow = get_bow(location)
    label = np.asmatrix(final_bow['DB:label']).transpose()
    final_bow = drop_words(final_bow)
    new_ds_without_label = final_bow.drop(['doc_id','DB:label','label_set','DB:loc'], axis=1)
    val = new_ds_without_label.values
    words = list(new_ds_without_label.columns)
    tf_transformer = TfidfTransformer()
    vectors_load = tf_transformer.fit_transform(val)
    mt = sc.sparse.csc_matrix(vectors_load.all())
    tfidf_mat = mt.todense()
    dataset = np.append(tfidf_mat, label, axis=1)
    return dataset

def retreive_file(train, floc, test_label):
    final_bow = get_bow(train)
    label = np.asmatrix(final_bow['DB:label']).transpose()
    final_bow = drop_words(final_bow)
    location = final_bow['DB:loc']
    new_ds_without_label = final_bow.drop(['DB:label','label_set','DB:loc'], axis=1)
    word_list = new_ds_without_label.columns
    val = new_ds_without_label.values
    words = list(new_ds_without_label.columns)
    tf_transformer = TfidfTransformer()
    vectors_load = tf_transformer.fit_transform(val)
    mt = sc.sparse.csc_matrix(vectors_load)
    tfidf_mat = mt.todense()
    train = np.append(tfidf_mat, label, axis=1)
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    df = pd.DataFrame(columns=word_list)
    counter = 0
    df=df.astype('float')
    with open(floc, 'r') as content_file:
        df.loc[counter]=0
        content = content_file.read()
        content=str(content).decode('UTF-8', 'ignore').lower()
        word_tokens = word_tokenize(content)
        filtered_sentence = [ps.stem(w) for w in word_tokens if not w in stop_words and not ps.stem(w) in stop_words and w.isalnum()]
        rt = {}
        for word in filtered_sentence:
            if word not in rt:
                rt[word]=0
            rt[word] += 1

        for word in rt:
            if word in df.columns:
                df.loc[counter, word] += rt[word]
    tf_transformer = TfidfTransformer()
    vectors_load = tf_transformer.fit_transform(df.values)
    mt = sc.sparse.csc_matrix(vectors_load)
    tfidf_mat = mt.todense()
    test = np.append(tfidf_mat, [[float(test_label)]], axis=1)
    test = test.astype('float')
    return np.asarray(train), np.asarray(test), location

def main():
    final_bow = get_bow('q2data/train')
    final_bow = final_bow.to_csv("q3_dataset.csv", sep=',', encoding='utf-8')
    label = np.asmatrix(final_bow['DB:label']).transpose()
    final_bow = drop_words(final_bow)
    final_bow.to_csv("q3_dataset_reduced.csv", sep=',', encoding='utf-8')
    new_ds_without_label = final_bow.drop(['doc_id','DB:label','label_set','DB:loc'], axis=1)
    val = new_ds_without_label.values
    words = list(new_ds_without_label.columns)
    tf_transformer = TfidfTransformer()
    vectors_load = tf_transformer.fit_transform(val)
    mt = sc.sparse.csc_matrix(vectors_load.all())
    tfidf_mat = mt.todense()
    dataset = np.append(tfidf_mat, label, axis=1)
    np.save('tfidf_reduced', sc.sparse.csc_matrix(dataset))
    pass

if __name__ == "__main__":
    main()
