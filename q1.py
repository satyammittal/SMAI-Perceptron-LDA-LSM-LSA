
# coding: utf-8

# In[3]:


import numpy as np


# In[54]:


val = np.array([[1,2,-1],[2,3,-1],[1,2,1]])


# In[220]:


def VotedPerceptron(array, epoch=1):
    n = array.shape[1]
    w = np.zeros(n-1)
    b = 0
    c = 0
    ans = []
    for t in range(epoch):
        for x in array:
            y_pred = np.dot(x[:-1],w) + b
            if y_pred * x[-1] > 0:
                c = c + 1
            elif c != 0:
                ret = (np.copy(w), b, c)
                ans.append(ret)
                w += x[:-1] * x[-1]
                b += x[-1]
                c = 1
            else:
                c += 1
                w += x[:-1] * x[-1]
                b += x[-1]
        ans.append((np.copy(w),b,c))
    return ans
    
    
def predictVoted(model, data):
    val = 0
    for m in model:
        val += m[2]*np.sign(np.dot(m[0],data)+m[1]-0.00001)
    return np.sign(val-0.00001)

model = VotedPerceptron(val)
predictVoted(model, [1,2])


# In[205]:


def VanillaPerceptron(array, epoch=1):
    n = array.shape[1]
    w = np.zeros(n-1)
    b = 0
    for t in range(epoch):
        for x in array:
            y_pred = np.dot(x[:-1],w) + b
            if y_pred * x[-1] > 0:
                pass
            else:
                w += x[:-1] * x[-1]
                b += x[-1]
    return (np.copy(w),b)
    
    
def predictVanilla(model, data):
    val = np.sign(np.dot(model[0],data)+model[1])
    return np.sign(val-0.00001)

model = VanillaPerceptron(val)
predictVanilla(model, [1,2])


# In[231]:


import pandas as pd
breast_ds = pd.read_csv("breast-cancer-wisconsin.csv", sep=',', header=None)


# In[232]:


breast_ds_pure = breast_ds.dropna(axis=0)
data = breast_ds_pure.iloc[:,1:]


# In[233]:


res = data.iloc[:,-1]
res = res - 3


# In[234]:


data.iloc[:,-1] = res


# In[238]:


breast_ds_arr = data.values.astype('int')


# In[241]:


breast_ds_arr


# In[102]:


ion = pd.read_csv("ionosphere.csv", sep=',', header=None)
ion_ds_pure = ion.dropna(axis=0)
data = ion_ds_pure.iloc[:,1:]


# In[104]:


data.iloc[:,-1] = data.iloc[:,-1].map({'b': -1, 'g': 1})


# In[108]:


ion_arr = data.values.astype('float')


# In[109]:


ion_arr


# In[113]:


#part 3


# In[119]:


#Ionosphere Dataset


# In[116]:


VotedPerceptron(ion_arr)


# In[118]:


VanillaPerceptron(ion_arr)


# In[120]:


#Breast Cancer Dataset


# In[121]:


VotedPerceptron(breast_ds_arr)


# In[122]:


VanillaPerceptron(breast_ds_arr)


# In[123]:


#Part 4


# In[124]:


min_epoch = 5
diff_epoch = 5
max_epoch = 50


# In[229]:


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
    total = len(test_real)
    same = 0
    for a in zip(test_real, test_pred):
        if a[0]==a[1]:
            same += 1
    return float(same*1.0/total*1.0)

def voted_perceptron_model(train, test, epoch=1):
    model = VotedPerceptron(train, epoch)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVoted(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    return score(test_real, test_out)


def vanilla_perceptron_model(train, test, epoch=1):
    model = VanillaPerceptron(train, epoch)
    test_in = np.delete(test, -1, axis=1)
    test_out = []
    for val in test_in:
        pred = predictVanilla(model, val)
        test_out.append(pred)
    test_real = test[:,-1]
    return score(test_real, test_out)
    

def perceptron_cross_valid(arr, string):
    k = 10
    for i in xrange(min_epoch, max_epoch+1, diff_epoch):
        split_arr = np.asarray(np.array_split(arr, k))
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

        


# In[230]:


print "--Ionosphere Dataset--"
print "Accuracy of Voted Perceptron"
perceptron_cross_valid(ion_arr, "voted")
print "Accuracy of Vanilla Perceptron"
perceptron_cross_valid(ion_arr, "vanilla")


# In[223]:


print "--Breast Cancer Dataset--"
print "Accuracy of Voted Perceptron"
perceptron_cross_valid(breast_ds_arr, "voted")
print "Accuracy of Vanilla Perceptron"
perceptron_cross_valid(breast_ds_arr, "vanilla")


# In[ ]:





# In[ ]:




