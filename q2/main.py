import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from LDA import *
from LSM import *

def plot_graph(vect, ax, color,label):
    n = 1000
    if len(vect)<3:
        vect.append(0)
    x = np.linspace(-4, 4, 10000)
    x = np.append(x,0)
    ax.plot(x, -1 * ( vect[0] * x +  vect[2])/vect[1], color=color, label=label)
    plt.legend(loc='best') 

def main():
    x = [[3,3], [3,0], [2,1], [0,2], [-1, 1], [0,0], [-1,-1], [1,0]]
    x = np.asarray(x)
    y = [1,1,1,1,-1,-1,-1,-1]
    y = np.asarray(y)
    lsa = least_squared_approach(x,y)
    fisher = fisher_lda(x,y)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    #figure = plt.figure()
    plt.scatter(x[:,0],x[:,1],c=y[:])
    plot_graph(lsa.tolist(),ax, "red", "LSA")
    plot_graph(fisher.tolist(), ax, "blue", "FISHER")
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    fig.set_size_inches(12, 8)
    plt.savefig("q2_a.jpg")
    plt.show()
    x = [[3,3], [3,0], [2,1], [0,1.5], [-1, 1], [0,0], [-1,-1], [1,0]]
    x = np.asarray(x)
    y = [1,1,1,1,-1,-1,-1,-1]
    y = np.asarray(y)
    lsa = least_squared_approach(x,y)
    fisher = fisher_lda(x,y)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.scatter(x[:,0],x[:,1],c=y[:])
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plot_graph(lsa.tolist(),ax, "red", "LSA")
    plot_graph(fisher.tolist(), ax, "blue", "FISHER")
    fig.set_size_inches(12, 8)
    plt.savefig("q2_b.jpg")
    plt.show()

if __name__ == "__main__":
    main()