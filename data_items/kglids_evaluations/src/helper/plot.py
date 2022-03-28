import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO: generalize it for all experiments
def visualize(exp_res: dict, exp_name: str, dataset: str):
    def plot_scores(exp_name, k: list, metric: list, j: list, d3l, aurum, d3l_j, aurum_j):
        default_ticks = range(len(k))
        plt.plot(default_ticks, metric, '--', color='g', label='KGLiDS', marker="x")
        plt.plot(default_ticks, j, '', color='g', label='KGLiDS+Join', marker="x")
        plt.plot(default_ticks, d3l, '--', color='cornflowerblue', label='D3L', marker="s")
        plt.plot(default_ticks, d3l_j, '', color='cornflowerblue', label='D3L+Join', marker="s")
        plt.plot(default_ticks, aurum, '--', color='darkorange', label='Aurum', marker="d")
        plt.plot(default_ticks, aurum_j, '', color='darkorange', label='Aurum+Join', marker="d")
        plt.xticks(default_ticks, k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K')
        plt.ylabel(exp_name)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        return plt

    scores_ap = pd.read_csv('../d3l_scores/attribute_precision_{}.csv'.format(dataset))
    d3l = scores_ap['D3L']
    d3l_j = scores_ap['D3L+J']
    aurum = scores_ap['Aurum']
    aurum_j = scores_ap['Aurum+J']
    # tus = scores_ap['TUS']

    k = []
    ap = []
    ap_j = []
    for key, v in exp_res.items():
        k.append(key)
        ap.append(v['attribute precision'])
        ap_j.append(v['attribute precision + J'])

    plt.figure(figsize=(7.5, 5.75))

    plot_scores(exp_name, k, ap, ap_j, d3l, aurum, d3l_j, aurum_j)
    plt.savefig('../plots/attribute_precision_{}.pdf'.format(dataset))
    plt.show()
