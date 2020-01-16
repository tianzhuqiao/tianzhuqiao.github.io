import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from common import *

def plot_tx():
    plt.clf()
    N = 2
    g = np.array([[0,0], [1,0], [2,0], [3,0]])
    plt.scatter(g[:,0], g[:,1])
    for i in range(2**N):
        plt.text(g[i,0]-0.1, g[i,1]+0.003, "b'"+np.binary_repr(i, N))
    plt.grid('on', ls=':')
    save_fig(plt, "../doc/image/mapping_bcd.svg")

    plt.clf()
    N = 2
    g = np.array([[0,0], [1,0], [3,0], [2,0]])
    plt.scatter(g[:,0], g[:,1])
    for i in range(2**N):
        plt.text(g[i,0]-0.1, g[i,1]+0.003, "b'"+np.binary_repr(i, N))
    plt.grid('on', ls=':')
    save_fig(plt, "../doc/image/mapping_gray.svg")

    plt.clf()
    N = 4
    g = gen_gray_table(N)
    plt.scatter(g[:,0], g[:,1])
    plt.plot(g[11,0], g[11,1], 'r.')
    for i in range(2**N):
        if i != 11:
            plt.text(g[i,0]+0.05, g[i,1], np.binary_repr(i, N))
    plt.text(g[11,0]+0.05, g[11,1], np.binary_repr(11, N), color='red')
    plt.xlim([-0.1,3.5])
    plt.grid('on', ls=':')
    plt.xlabel('group 1')
    plt.ylabel('group 2')
    save_fig(plt, "../doc/image/mapping_gray_table.svg")
