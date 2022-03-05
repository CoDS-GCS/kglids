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

def visualize(r1: dict, r2, r3, r4):
    def plot_scores(k: list, r1: list, j1: list, metric_name: str, r2, j2, d3l, d3l_j, r3, j3, r4, j4):
        print(j1)
        print(j3)
        print(j4)
        default_ticks = range(len(k))
        plt.plot(default_ticks, r1, 'g', label='KGLiDS (semantic similarity = 0.5, 7 hops)', marker="x")
        plt.plot(default_ticks, j1, 'limegreen', label='KGLiDS+Join (pkfk thresh: 0.60 | 5 hops)', marker="x")
        #plt.plot(default_ticks, r2, 'gray', label='KGLiDS', marker="s")
        #plt.plot(default_ticks, j2, 'black', label='KGLiDS+Join (pkfk thresh: 0.80 | 5 hops)', marker="s")
        plt.plot(default_ticks, d3l, 'cornflowerblue', label='D3L', marker="d")
        plt.plot(default_ticks, d3l_j, 'blue', label='D3L+Join', marker="d")
        plt.plot(default_ticks, j3, 'blue', label='D3L+Join', marker="d")
        plt.plot(default_ticks, j4, 'blue', label='D3L+Join', marker="d")
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

    scores_coverage = pd.read_csv('../../kglids_evaluations/d3l_scores/coverage_smallerReal.csv')
    d3l = scores_coverage['D3L']
    d3l_j = scores_coverage['D3L+J']
    # aurum = scores_coverage['Aurum']
    # aurum_j = scores_coverage['Aurum+J']
    # tus = scores_coverage['TUS']


    k = []
    coverage1 = []
    coveragej1 = []
    for key, v in r1.items():
        k.append(key)
        coverage1.append(v['coverage'])
        coveragej1.append(v['coverage+J'])

    coverage2 = []
    coveragej2 = []
    for key, v in r2.items():
        coverage2.append(v['coverage'])
        coveragej2.append(v['coverage+J'])

    coverage3 = []
    coveragej3 = []
    for key, v in r3.items():
        k.append(key)
        coverage3.append(v['coverage'])
        coveragej3.append(v['coverage+J'])

    coverage4 = []
    coveragej4 = []
    for key, v in r4.items():
        coverage4.append(v['coverage'])
        coveragej4.append(v['coverage+J'])



    plt.figure(figsize=(10, 6))
    plot_scores(k, coverage1, coveragej1, 'Target coverage', coverage2, coveragej2, d3l, d3l_j, coverage3, coveragej3, coverage4, coveragej4)
    plt.savefig('0.75 vs 0.5 target coverage.pdf')
    plt.show()

def main():
    res1 = load_cache('060_target_coverage_0.75_5_hops_smallerReal_k-260')
    res2 = load_cache('080_target_coverage_0.75_5_hops_smallerReal_k-260')
    res3 = load_cache('050_target_coverage_0.75_5_hops_smallerReal_k-260')
    res4 = load_cache('050_target_coverage_0.75_7_hops_smallerReal_k-260')

    visualize(res1, res2, res3, res4)

main()