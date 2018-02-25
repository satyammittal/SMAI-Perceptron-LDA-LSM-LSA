import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
from func import *
from VanillaPerceptron import *
from VotedPerceptron import *
random.seed(9001)
eps = sys.float_info.epsilon

min_epoch = 5
diff_epoch = 5
max_epoch = 50

def read_breast_ds():
    breast_ds = pd.read_csv("breast-cancer-wisconsin.csv", sep=',', header=None)
    col = breast_ds.columns
    for i in col:
        breast_ds[i].replace('?', None, inplace=True)
    breast_ds_pure = breast_ds.dropna(axis=1, how='any')
    data = breast_ds_pure.iloc[:,1:]
    res = data.iloc[:,-1]
    res = res - 3
    data.iloc[:,-1] = res
    breast_ds_arr = data.values.astype('int')
    return breast_ds_arr

def read_ionosphere():
    ion = pd.read_csv("ionosphere.csv", sep=',', header=None)
    col = ion.columns
    for i in col:
        ion[i].replace('?', None, inplace=True)
    ion_ds_pure = ion.dropna(axis=0)
    data = ion_ds_pure.iloc[:,1:]
    data.iloc[:,-1] = data.iloc[:,-1].map({'b': -1, 'g': 1})
    ion_arr = data.values.astype('float')
    return ion_arr

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
        print ( ("Epoch {0} -> {1} +/- {2}").format(i, larr.mean(), 2*larr.std()) )
        acc_mat.append(larr.mean())
    return acc_mat

def main():
    ion_arr = read_ionosphere()
    breast_ds_arr = read_breast_ds()
    print "--Ionosphere Dataset--"
    print "Accuracy of Voted Perceptron"
    ion_acc_voted = perceptron_cross_valid(ion_arr, "voted")
    print "Accuracy of Vanilla Perceptron"
    ion_acc_vanilla = perceptron_cross_valid(ion_arr, "vanilla")
    print "--Breast Cancer Dataset--"
    print "Accuracy of Voted Perceptron"
    breast_acc_voted = perceptron_cross_valid(breast_ds_arr, "voted")
    print "Accuracy of Vanilla Perceptron"
    breast_acc_vanilla = perceptron_cross_valid(breast_ds_arr, "vanilla")
    epochs = [i for i in xrange(min_epoch, max_epoch+1, diff_epoch)]
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    plt.subplot(211)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs,breast_acc_voted , '.r-', label='Voted')
    plt.plot(epochs,breast_acc_vanilla , '.b-', label='Vanilla')
    plt.legend(loc='upper left')
    plt.title('Perceptron of Breast Cancer Dataset')
    plt.subplot(212)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs,ion_acc_voted , '.r-', label='Voted')
    plt.plot(epochs,ion_acc_vanilla , '.b-', label='Vanilla')
    plt.legend(loc='upper right')
    plt.title('Perceptron of Ionosphere Dataset')
    plt.show()
    #fig.savefig('q1_ion.png')
    pass

if __name__ == "__main__":
    main()