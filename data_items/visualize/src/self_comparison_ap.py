import os

import tqdm
import pickle
import random
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_cache(load_as='cache'):
    with open('../../kglids_evaluations/cache/' + load_as + '.pkl', 'rb') as handle:
        return pickle.load(handle)

def visualize(r1: dict, r2, r3, r4, r5):
    def plot_scores(k: list, r1: list, j1: list, metric_name: str, r2, j2, d3l, d3l_j, r3, j3, r4, j4, r5, j5):
        print(j1)
        print(j4)
        default_ticks = range(len(k))
        plt.plot(default_ticks, r1, 'g', label='KGLiDS', marker="x")
        plt.plot(default_ticks, j1, 'limegreen', label='KGLiDS+Join (pkfk thresh: 0.60 | 5 hops)', marker="x")
        # plt.plot(default_ticks, r2, 'cornflowerblue', label='KGLiDS', marker="s")
        plt.plot(default_ticks, j2, 'blue', label='KGLiDS+Join (pkfk thresh: 0.70 | 5 hops)', marker="s")
        plt.plot(default_ticks, d3l, 'cornflowerblue', label='D3L', marker="d")
        plt.plot(default_ticks, d3l_j, 'blue', label='D3L+Join', marker="d")
        #plt.plot(default_ticks, r4, 'gray', label='KGLiDS', marker="^")
        plt.plot(default_ticks, j3, 'yellow', label='KGLiDS+Join (pkfk thresh: 0.80 | 5 hops)', marker="^")
        plt.plot(default_ticks, j4, 'red', label='KGLiDS+Join (pkfk thresh: 0.50 | 5 hops)', marker="^")
        plt.plot(default_ticks, j5, 'black', label='KGLiDS+Join (pkfk thresh: 0.50 | 7 hops)', marker="^")
        #plt.plot(default_ticks, tus, 'gray', label='TUS', marker="^")
        plt.xticks(default_ticks, k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K')
        plt.ylabel(metric_name)
        # plt.title(metric_name)
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.tight_layout()
        plt.grid()
        return plt

    scores_coverage = pd.read_csv('../../kglids_evaluations/d3l_scores/attribute_precision_smallerReal.csv')
    d3l = scores_coverage['D3L']
    d3l_j = scores_coverage['D3L+J']
    # aurum = scores_coverage['Aurum']
    # aurum_j = scores_coverage['Aurum+J']
    # tus = scores_coverage['TUS']

    k = []
    ap1 = []
    ap_j1 = []
    for key, v in r1.items():
        k.append(key)
        ap1.append(v['attribute precision'])
        ap_j1.append(v['attribute precision + J'])


    ap2 = []
    ap_j2 = []
    for key, v in r2.items():
        ap2.append(v['attribute precision'])
        ap_j2.append(v['attribute precision + J'])



    ap3 = []
    ap_j3 = []
    for key, v in r3.items():
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


    plt.figure(figsize=(10, 6))
    plot_scores(k, ap1, ap_j1, 'Attribute precision', ap2, ap_j2, d3l, d3l_j, ap3, ap_j3, ap4, ap_j4, ap5, ap_j5)
    #plt.savefig('0.75 vs 0.5 attribute precision.pdf')
    plt.show()

def main():
    print(os.path.exists('../../kglids_evaluations'))
    res0 = load_cache('060_attribute_precision_0.75_7_hops_smallerReal_k-260')
    res1 = load_cache('060_attribute_precision_0.75_5_hops_smallerReal_k-260')
    res2 = load_cache('attribute_precision_0.75_7_hops_smallerReal_k-260')
    res3 = load_cache('080_attribute_precision_0.75_5_hops_smallerReal_k-260')
    res4 = load_cache('050_attribute_precision_0.75_5_hops_smallerReal_k-260')
    res5 = load_cache('050_attribute_precision_0.75_7_hops_smallerReal_k-260')
    print(res0)
    visualize(res1, res2, res3, res4, res5)

main()