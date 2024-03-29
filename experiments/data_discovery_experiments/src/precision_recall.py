import tqdm
import time
import random
import numpy as np
from matplotlib import pyplot as plt
import pickle

from helper.queries import *
from helper.config import connect_to_stardog
from helper.cache import cache_score
from helper.common import load_groundtruth, get_table_mapping

# *************EXPERIMENT PARAMETERS**************
THRESHOLD = 1
NO_RANDOM_QUERY_TABLES = 100
DATASET = 'smallerReal'   # synthetic
DATABASE = 'smaller_real_fine_grained_05' # 'kglids_smaller_real_fine_grained_075'  
MAX_K = 185 if DATASET == 'smallerReal' else 350
# ************************************************
EXPERIMENT_NAME = 'precision_recall'
SAVE_RESULT_AS = EXPERIMENT_NAME + '_' + DATASET + '_' + DATABASE
SPARQL = connect_to_stardog(db=DATABASE)
# ************************************************


def load_cache(load_as='cache'):
    with open('../cache/' + load_as, 'rb') as handle:
        return pickle.load(handle)





def get_n_random_tables(df: pd.DataFrame, n: int):
    print("getting {} random tables".format(n))
    query_tables = list(np.unique(df[df.columns[0]].to_list()))
    return random.sample(query_tables, n)


def calculate_scores(pred: list, test: list):
    tp = fp = 0
    
    if not pred:
        return 0, 0
    
    for pair in pred:
        if pair in test:
            tp = tp + 1  # have it ground truth and have it in predictions
        else:
            fp = fp + 1  # do not have it ground truth but have it in predictions

    test2 = [i for i in test if i[0] == pred[0][0]]
    fn = len([i for i in test2 if i not in pred])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def run_experiment(df, test_mapping: list):
    query_tables = get_n_random_tables(df, NO_RANDOM_QUERY_TABLES)
    
    print('\n• running experiment!')
    top_k = []
    if DATASET == 'smallerReal':
        top_k = [5, 20, 35, 50, 65, 80, 95, 110, 125, 140, 155, 170, 185]
    elif DATASET == 'synthetic':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350]
    
    precisions = {}
    recalls = {}
    for table in tqdm.tqdm(query_tables):
        top_related_tables = get_top_k_related_tables(SPARQL, table, max(top_k), THRESHOLD)
        precision_per_table = []
        recall_per_table = []

        for k in top_k:
            precision, recall = calculate_scores(top_related_tables[:k], test_mapping)
            precision_per_table.append(precision)
            recall_per_table.append(recall)
        precisions[table] = precision_per_table
        recalls[table] = recall_per_table

    precisions_df = pd.DataFrame.from_dict(precisions, orient='index', columns=top_k)
    recalls_df = pd.DataFrame.from_dict(recalls, orient='index', columns=top_k)
    print('Average Precisions:\n', precisions_df.mean())
    print('Average Recalls:\n', recalls_df.mean())

    res = {k: {'precision': np.mean(precisions_df[k]), 'recall': np.mean(recalls_df[k])} for k in top_k}
    cache_score(res, max(top_k), top_k, SAVE_RESULT_AS)


def visualize(exp_res: dict):
    def plot_scores(top_k: list, metric: list, metric_name: str, title, d3l, aurum):
        label_size = 17
        default_ticks = range(len(top_k))
        plt.plot(default_ticks, d3l, 'cornflowerblue', linestyle='--', label='$D^3L$', marker="o")
        plt.plot(default_ticks, aurum, 'darkorange', linestyle='--', label='Aurum', marker="x")
        plt.plot(default_ticks, metric, 'forestgreen', label='KGLiDS', marker="s")
        plt.xticks(default_ticks, top_k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K', fontsize=label_size)
        plt.ylabel(metric_name, fontsize=label_size)
        plt.title(title, y=-0.20, fontsize=label_size)
        plt.legend(loc='lower right')
        plt.grid()
        return plt

    scores_precision = pd.read_csv('../d3l_scores/precision_{}.csv'.format(DATASET))
    scores_recall = pd.read_csv('../d3l_scores/recall_{}.csv'.format(DATASET))
    d3l_precision = scores_precision['D3L']
    d3l_recall = scores_recall['D3L']
    aurum_precision = scores_precision['Aurum']
    aurum_recall = scores_recall['Aurum']

    k = []
    precision = []
    recall = []
    for key, v in exp_res.items():
        k.append(key)
        precision.append(v['precision'])
        recall.append(v['recall'])

    scores_precision['KGLids'] = precision
    scores_recall['KGLids'] = recall

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    _ = plot_scores(k, precision, 'Precision', '(a)', d3l_precision, aurum_precision)
    plt.subplot(1, 2, 2)
    _ = plot_scores(k, recall, 'Recall', '(b)', d3l_recall, aurum_recall)
    plt.tight_layout()
    plt.savefig('../plots/{}.pdf'.format(SAVE_RESULT_AS), dpi=300)
    plt.show()


def main():
    
    df = load_groundtruth(DATASET)
    test_mapping = get_table_mapping(df)
    
    t1 = time.time()
    run_experiment(df, test_mapping)
    print('Total time taken: ', time.time()-t1)

    exp_res = load_cache(f'{SAVE_RESULT_AS}_k-{MAX_K}.pkl')
    visualize(exp_res)
    print('done.')


main()
