import os

import pylab as pl
import tqdm
import pickle
import random
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_cache(load_as='cache'):
    with open('../cache/' + load_as + '.pkl', 'rb') as handle:
        return pickle.load(handle)


def visualize(r1, r2):
    def plot_scores(k: list, j1: list, metric_name: str, r2, j2):
        label_size = 15
        default_ticks = range(len(k))
        plt.plot(default_ticks, j2, color='g', label='KGLiDS+Join (with Deep embeddings)', marker="x")
        plt.plot(default_ticks, j1, color='black', label='KGLiDS+Join (without Deep embeddings)', marker="x")
        plt.plot(default_ticks, r2, '--', color='g', label='KGLiDS', marker="x")
        # plt.plot(default_ticks, r2, 'cornflowerblue', label='KGLiDS', marker="s")
        # plt.plot(default_ticks, d3l, 'cornflowerblue', label='D3L', marker="d")
        # plt.plot(default_ticks, d3l_j, 'blue', label='D3L+Join', marker="d")
        # # plt.plot(default_ticks, r4, 'gray', label='KGLiDS', marker="^")
        # plt.plot(default_ticks, j3, 'black', label='KGLiDS+Join (Deep embeddings = 0.90)', marker="^")
        # plt.plot(default_ticks, j4, 'red', label='KGLiDS+Join (pkfk thresh: 0.50 | 5 hops)', marker="^")
        # plt.plot(default_ticks, j5, 'black', label='KGLiDS+Join (pkfk thresh: 0.50 | 7 hops)', marker="^")
        # plt.plot(default_ticks, tus, 'gray', label='TUS', marker="^")
        plt.xticks(default_ticks, k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K', fontsize=label_size)
        plt.ylabel(metric_name, fontsize=label_size)
        # plt.title(metric_name)
        # plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.grid()
        pl.title('(b)', y=-0.20, fontsize=label_size)
        return plt

    scores_coverage = pd.read_csv('../d3l_scores/attribute_precision_smallerReal.csv')
    d3l = scores_coverage['D3L']
    d3l_j = scores_coverage['D3L+J']
    # aurum = scores_coverage['Aurum']
    # aurum_j = scores_coverage['Aurum+J']
    # tus = scores_coverage['TUS']
    THRESH = 260
    k = []
    ap1 = []
    ap_j1 = []
    for key, v in r1.items():
        if key <= THRESH:
            k.append(key)
            ap1.append(v['attribute precision'])
            ap_j1.append(v['attribute precision + J'])

    ap2 = []
    ap_j2 = []
    for key, v in r2.items():
        if key <= THRESH:
            ap2.append(v['attribute precision'])
            ap_j2.append(v['attribute precision + J'])

    '''
    ap3 = []
    ap_j3 = []
    for key, v in r3.items():
        if key <= THRESH:
            ap3.append(v['attribute precision'])
            ap_j3.append(v['attribute precision + J'])

    ap4 = []
    ap_j4 = []
    for key, v in r4.items():
        ap4.append(v['attribute precision'])
        ap_j4.append(v['attribute precision + J'])
    ap5 = []
    ap_j5 = []
    for key, v in r5.items():
        ap5.append(v['attribute precision'])
        ap_j5.append(v['attribute precision + J'])
    '''
    # d3l = d3l[:len(k)].tolist()
    # d3l_j = d3l_j[:len(k)].tolist()
    # print(ap1, "\n", ap2, "\n", ap3, "\n", d3l, "\n", d3l_j)

    # plt.figure(figsize=(7.5, 5.75))
    return plot_scores(k, ap_j1, 'Attribute precision', ap2, ap_j2)  # d3l, d3l_j)  # , ap3, ap_j3)  # , ap4, ap_j4, ap5, ap_j5)
    # plt.savefig('../plots/improvements_with_deep_embeddings.pdf')
    # plt.show()


def plot_comparison():
    res1 = load_cache('attribute_precision_smallerReal_k-260_without_de')
    res2 = load_cache('attribute_precision_smallerReal_k-260')
    return visualize(res1, res2)

