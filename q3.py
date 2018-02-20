
# coding: utf-8

# In[1]:


import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


# In[2]:


def get_data(loc):
    ans = {}
    dirs = os.listdir(loc)
    for ind in dirs:
        doc = []
        files = os.listdir(loc+'/'+ind)
        for f in files:
            floc = loc+'/'+ind+'/'+f
            with open(floc, 'r') as content_file:
                content = content_file.read()
                content=str(content).decode('UTF-8', 'ignore')
                doc.append(content)
        ans[ind]=doc
    return ans
val = get_data('q2data/train')


# In[ ]:





# In[12]:


vectorizer = TfidfVectorizer(min_df=2,max_features= 1000,analyzer='word',stop_words='english')
print val.keys()
print "assd"
for d in val.keys():
    docs = val[d]
    # Train the vectorizer on the descriptions
    vectorizer = vectorizer.fit(docs)
for d in val.keys():
    docs = val[d]
    # Convert descriptions to feature vectors
    tfidf = vectorizer.transform(docs)
    tfidf=tfidf.todense()
    tfidf=tfidf.tolist()
    print tfidf[0]
    #print X_tfidf


# In[2]:


stop_words = set(stopwords.words('english'))


# In[18]:


def get_bow(loc):
    ans = {}
    dirs = os.listdir(loc)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    df = pd.DataFrame(columns=['label'])
    counter=0
    for ind in dirs:
        files = os.listdir(loc+'/'+ind)
        for f in files:
            df.loc[counter]=0
            df.loc[counter, 'label'] = ind
            floc = loc+'/'+ind+'/'+f
            with open(floc, 'r') as content_file:
                content = content_file.read()
                content=str(content).decode('UTF-8', 'ignore').lower()
                word_tokens = word_tokenize(content)
                filtered_sentence = [ps.stem(w) for w in word_tokens if not w in stop_words and not ps.stem(w) in stop_words and w.isalnum()]
                for word in filtered_sentence:
                    if word not in df.columns:
                        df[word]=0
                    df.loc[counter, word] += 1
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


# In[4]:


final_bow = get_bow('q2data/train')


# In[6]:


final_bow.to_csv("q3_bow.csv", sep=',', encoding='utf-8')


# In[7]:


len(final_bow.columns)


# In[8]:


copy_bow = final_bow.copy()


# In[9]:


copy_bow['label']=0


# In[15]:


copy_bow = copy_bow.drop('doc_id', axis=1)


# In[ ]:





# In[19]:


add_labels('q2data/train', dr)


# In[20]:


dr.to_csv("q3_bow_final.csv", sep=',', encoding='utf-8')


# In[7]:


new_ds_without_label = dr.drop('label', axis=1)


# In[19]:


val = new_ds_without_label.values
words = list(new_ds_without_label.columns)


# In[20]:


tf_transformer = TfidfTransformer()


# In[10]:


print "ass"


# In[11]:


dr = pd.read_csv("q3_bow_final.csv", sep=',', encoding='utf-8')


# In[12]:


label = np.asmatrix(dr['label']).transpose()


# In[21]:


dr = dr.drop('docid', axis=1)


# In[1]:


tf_transformer = TfidfTransformer()
r = tf_transformer.fit_transform(val)
np.save('tfidf_1', r) 


# In[2]:


vectors_load = np.load('tfidf.npy')


# In[3]:


vectors_load


# In[4]:


mt = sc.sparse.csc_matrix(vectors_load.all())


# In[5]:


mt.shape


# In[6]:


tfidf_mat = mt.todense()


# In[7]:


tfidf_mat[0]


# In[53]:


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
        clas = k+1
        print np.unique(label)
        arr = np.array([makeitsingular(t, clas) for t in label])
        array[:, -1] = arr
        w.append(VanillaPerceptron(array, epoch))
    return w
    
    
def predictVanilla(model, data):
    y = data.copy()
    y = np.append(y,1)
    minh = -10000000000007
    pred = 1
    counter = 0
    for m in model:
        counter += 1
        val = np.dot(m,y)
        if minh < val:
            minh = val
            pred = counter
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
    model = multiclassPerceptron(train, epoch, 5)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVanilla(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    print test_real, test_out
    return score(test_real, test_out)

def sign(num):
    if num<=0:
        return -1
    else:
        return 1
    
def perceptron_cross_valid(arr, string):
    k = 10
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
            if string=="voted":
                accuracy = voted_perceptron_model(train_arr, test_arr, i)
            elif string=="vanilla":
                accuracy = vanilla_perceptron_model(train_arr, test_arr, i)
            ans_arr.append(accuracy)
        larr = np.array(ans_arr)
        print ("Epoch {0} -> {1} +/- {2}").format(i, larr.mean(), 2*larr.std())
        acc_mat.append(larr.mean())
    return acc_mat


# In[13]:


dataset = np.append(tfidf_mat, label, axis=1)


# In[14]:


dataset.shape


# In[54]:


model = perceptron_cross_valid(np.asarray(dataset), "vanilla")


# In[76]:


np.asarray(dataset)[:,1000]


# In[48]:


w = multiclassPerceptron(np.asarray(dataset).copy(),1, 5)


# In[55]:


model


# In[ ]:




