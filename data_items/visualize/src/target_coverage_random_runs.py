import pickle
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


def load_all_cache():
    res = []
    files = os.listdir("../cache")
    for f in files:
        with open("../cache/" + f, 'rb') as handle:
            res.append(pickle.load(handle))
    return res


def visualize(res: list):
    def plot_scores(k: list, exp_name: str, all_runs: list, d3l):
        default_ticks = range(len(k))

        plt.plot(default_ticks, all_runs[0], 'g', label='KGLiDS run #1', marker=".")
        plt.plot(default_ticks, all_runs[1], 'b', label='KGLiDS run #2', marker="+")
        plt.plot(default_ticks, all_runs[2], 'aqua', label='KGLiDS run #3', marker="^")
        plt.plot(default_ticks, all_runs[3], 'y', label='KGLiDS run #4', marker="*")
        plt.plot(default_ticks, all_runs[4], 'deeppink', label='KGLiDS run #5', marker="|")
        plt.plot(default_ticks, all_runs[5], 'red', label='KGLiDS run #6', marker="2")
        plt.plot(default_ticks, all_runs[6], 'gray', label='KGLiDS run #7', marker="o")
        plt.plot(default_ticks, all_runs[7], 'orange', label='KGLiDS run #8', marker="s")
        plt.plot(default_ticks, all_runs[8], 'brown', label='KGLiDS run #9', marker="d")
        plt.plot(default_ticks, all_runs[9], 'purple', label='KGLiDS run #10', marker="h")
        plt.plot(default_ticks, d3l, 'black', label='D3l', marker="P")

        plt.xticks(default_ticks, k)
        plt.title("Target coverage random runs without join path")
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K')
        plt.ylabel(exp_name)
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.tight_layout()
        plt.grid()
        return plt

    scores_coverage = pd.read_csv('../../kglids_evaluations/d3l_scores/coverage_smallerReal.csv')
    d3l = scores_coverage['D3L']


    k = list(res[0].keys())
    print(k)
    all_coverage_runs= []

    for r in res:
        run_n = []
        for v in list(r.values()):
            run_n.extend(list(v.values()))
        all_coverage_runs.append(run_n)

    plt.figure(figsize=(7.5, 5))
    plot_scores(k, 'Target coverage', all_coverage_runs, d3l)
    plt.savefig('n_runs.jpg')
    plt.show()


def main():
    res_random_runs = load_all_cache()
    print(res_random_runs)
    visualize(res_random_runs)


main()