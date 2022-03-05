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


def visualize(res: dict):
    scores_coverage = pd.read_csv('../../kglids_evaluations/d3l_scores/coverage_smallerReal.csv')
    d3l = scores_coverage['D3L']

    k = list(res.keys())
    k = list(map(lambda x: str(x), k))
    print('Top-k: ', k)
    std = []
    mean = []
    for keys in res.keys():
        std.append(res[keys]['std'])
        mean.append(res[keys]['mean'])

    scores_coverage = pd.read_csv('../../kglids_evaluations/d3l_scores/coverage_smallerReal.csv')
    d3l = scores_coverage['D3L']
    print('mean: ', mean)
    print('std: ', std)
    plt.bar(k, mean, width=0.6, yerr=std, color="navy", ecolor='red', capsize=6)
    plt.plot(k, mean, 'red', label='KGLiDS mean', marker=".")
    plt.plot(k, d3l, 'black', label='D3l', marker="P")
    plt.xlabel("K")
    plt.ylabel("Target coverage")
    plt.title("Standard deviation & Mean - Target coverage without join path")
    plt.legend()
    plt.grid()
    plt.savefig('std.jpg')
    plt.show()


def get_mean_and_std(res: dict):
    for k in res.keys():
        loc = {'mean': np.mean(res[k]), 'std': np.std(res[k]), 'values': res[k]}
        res[k] = loc

    return res


def custom_format(res: list):
    all_coverage_runs = []
    K = list(res[0].keys())
    for r in res:
        run_n = []
        for v in list(r.values()):
            run_n.extend(list(v.values()))
        all_coverage_runs.append(run_n)

    format_result = {}
    for k in range(0, len(K)):
        coverage_per_k = []
        for runs in all_coverage_runs:
            coverage_per_k.append(runs[k])
        format_result[K[k]] = coverage_per_k

    return format_result


def main():
    res_random_runs = load_all_cache()
    res_random_runs = custom_format(res_random_runs)
    res_random_runs = get_mean_and_std(res_random_runs)
    visualize(res_random_runs)


main()
