import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(ks, kglids_scores, santos_scores, d3l_scores, metric_name):
    label_size = 17
    
    plt.plot(ks, kglids_scores, color='g', label='KGLiDS', marker="s")
    plt.plot(ks, santos_scores, linestyle='--', color='cornflowerblue', label='SANTOS', marker="d")
    plt.plot(ks, d3l_scores, linestyle='--', color='darkorange', label='$D^3L$', marker="o")
    plt.xticks(ks)
    min_y = max(round(min(kglids_scores+santos_scores+d3l_scores), 1)-0.1, 0)
    print(min_y)
    plt.ylim(ymin=min_y)
    plt.yticks(np.arange(min_y, 1.1, 0.1))
    plt.xlabel('k', fontsize=label_size)
    plt.ylabel(metric_name, fontsize=label_size)
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    input_file = 'santos_scores.csv'
    df = pd.read_csv(input_file)
    
    plot(df['k'], df['kglidsPrecision'].tolist(), df['SantosPrecision'].tolist(), df['d3lPrecision'].tolist(), 'Precision@k')
    plot(df['k'], df['kglidsRecall'].tolist(), df['SantosRecall'].tolist(), df['d3lRecall'].tolist(), 'Recall@k')

if __name__ == '__main__':
    main()