import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.legend import _get_legend_handles_labels


def plot(ax, ks, metric_name, title, colors, markers, line_styles, min_y=None, max_y=None, **kwargs):
    title_size = 14
    label_size = 12
    
    
    for color, marker, line_style, (k, v) in zip(colors, markers, line_styles, kwargs.items()):
        ax.plot(ks, v, color=color, linestyle=line_style, label=k, marker=marker)

    ax.set_xticks(ks)
    min_y = min_y or max(round(min(list(itertools.chain.from_iterable(kwargs.values()))), 1)-0.1, 0)
    max_y = max_y or 1.025
    print(min_y)
    ax.set_ylim(ymin=min_y, ymax=max_y)
    ax.set_yticks(np.arange(min_y, max_y, 0.1))
    ax.set_xlabel('k', fontsize=label_size)
    ax.set_ylabel(metric_name, fontsize=label_size)
    ax.set_title(title, fontsize=title_size, pad=10)
    ax.grid()



def main():
    input_file = 'santos_scores.csv'
    santos_scores_df = pd.read_csv(input_file)
    santos_k = santos_scores_df['k']
    kglids_santos_precision = santos_scores_df['kglidsPrecision'].tolist()
    santos_santos_precision = santos_scores_df['SantosPrecision'].tolist()
    starmie_santos_precision = santos_scores_df['starmiePrecision'].tolist()
    kglids_santos_recall = santos_scores_df['kglidsRecall'].tolist()
    santos_santos_recall = santos_scores_df['SantosRecall'].tolist()
    starmie_santos_recall = santos_scores_df['starmieRecall'].tolist()
    d3l_santos_precision = santos_scores_df['d3lPrecision'].tolist()
    d3l_santos_recall = santos_scores_df['d3lRecall'].tolist()
    
    smaller_real_precision_df = pd.read_csv('../../d3l_scores/precision_smallerReal.csv')
    smaller_real_recall_df = pd.read_csv('../../d3l_scores/recall_smallerReal.csv')
    smaller_real_k = smaller_real_recall_df['k']
    kglids_smaller_real_precision = []
    santos_smaller_real_precision = smaller_real_precision_df['SANTOS']
    starmie_smaller_real_precision = smaller_real_precision_df['Starmie']
    kglids_smaller_real_recall = []
    santos_smaller_real_recall = smaller_real_recall_df['SANTOS']
    starmie_smaller_real_recall = smaller_real_recall_df['Starmie']
    d3l_smaller_real_precision = smaller_real_precision_df['D3L']
    aurum_smaller_real_precision = smaller_real_precision_df['Aurum']
    d3l_smaller_real_recall = smaller_real_recall_df['D3L']
    aurum_smaller_real_recall = smaller_real_recall_df['Aurum']
    
    tus_scores_df = pd.read_csv('tus_scores.csv')
    tus_k = tus_scores_df['k']
    kglids_tus_precision = tus_scores_df['kglidsPrecision'].tolist()
    santos_tus_precision = tus_scores_df['SantosPrecision'].tolist()
    starmie_tus_precision = tus_scores_df['starmiePrecision'].tolist()
    kglids_tus_recall = tus_scores_df['kglidsRecall'].tolist()
    santos_tus_recall = tus_scores_df['SantosRecall'].tolist()
    starmie_tus_recall = tus_scores_df['starmieRecall'].tolist()
    
    with open('../../cache/precision_recall_smallerReal_k-185.pkl', 'rb') as f:
        scores_dict = pickle.load(f)
    for k, v in scores_dict.items():
        kglids_smaller_real_precision.append(v['precision'])
        kglids_smaller_real_recall.append(v['recall'])

    # plt.figure()
    # fig, ((ax3, ax5, ax1), (ax4, ax6, ax2)) = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    fig, ((ax3, ax4), (ax5, ax6), (ax1, ax2)) = plt.subplots(3, 2, figsize=(8, 12), layout='constrained')
    colors_santos = ['orange', 'crimson', 'forestgreen']
    markers_santos = ['x', 'o', 's']
    line_styles_santos = ['--', '--', '-']
    plot(ax1, santos_k, 'Precision@k', 'Precision SANTOS Small', colors_santos, markers_santos, line_styles_santos, 
         Starmie=starmie_santos_precision, SANTOS=santos_santos_precision,
         KGLiDS=kglids_santos_precision, min_y=0.5)
    plot(ax2, santos_k, 'Recall@k', 'Recall SANTOS Small', colors_santos, markers_santos, line_styles_santos,
         Starmie=starmie_santos_recall, SANTOS=santos_santos_recall,
        KGLiDS=kglids_santos_recall)

    colors_smaller_real = ['orange', 'crimson', 'forestgreen']
    makrers_smaller_real = ['x', 'o', 's']
    line_styles_smaller_real = ['--', '--', '-']
    plot(ax3, smaller_real_k, 'Precision@k', 'Precision $D^3L$ Small', colors_smaller_real, makrers_smaller_real, line_styles_smaller_real, 
         Starmie=starmie_smaller_real_precision, SANTOS=santos_smaller_real_precision,
          KGLiDS=kglids_smaller_real_precision)
    plot(ax4, smaller_real_k, 'Recall@k', 'Recall $D^3L$ Small', colors_smaller_real, makrers_smaller_real, line_styles_smaller_real, 
         Starmie=starmie_smaller_real_recall, SANTOS=santos_smaller_real_recall,
         KGLiDS=kglids_smaller_real_recall)
    
    colors_tus = ['orange', 'crimson', 'forestgreen']
    makrers_tus = ['x', 'o', 's']
    line_styles_tus = ['--', '--', '-']
    plot(ax5, tus_k, 'Precision@k', 'Precision TUS Small', colors_tus, makrers_tus, line_styles_tus, 
         Starmie=starmie_tus_precision, SANTOS=santos_tus_precision,
          KGLiDS=kglids_tus_precision, min_y=0.5)
    plot(ax6, tus_k, 'Recall@k', 'Recall TUS Small', colors_tus, makrers_tus, line_styles_tus, 
         Starmie=starmie_tus_recall, SANTOS=santos_tus_recall,
         KGLiDS=kglids_tus_recall, max_y=0.55)
    # plt.legend()
    handles, labels = _get_legend_handles_labels(fig.axes)
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            if label == 'KGLiDS':
                unique_labels.insert(0, label)
                unique_handles.insert(0, handle)
            else:
                unique_labels.append(label)
                unique_handles.append(handle)
    fig.legend(unique_handles, unique_labels, loc='outside upper center', ncol=4, fontsize=14)
    # fig.tight_layout()
    plt.savefig('smaller_real_and_santos_and_tus_precision_recall_column.pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()