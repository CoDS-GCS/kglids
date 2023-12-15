import argparse

import tqdm
import time
import random
import numpy as np
from matplotlib import pyplot as plt
import pickle

from helper.queries import *
from helper.config import connect_to_graphdb
from helper.cache import cache_score
from helper.common import load_groundtruth, get_table_mapping

# *************EXPERIMENT PARAMETERS**************
THRESHOLD = 0.9
NO_RANDOM_QUERY_TABLES = 100
DATASET = 'TUS' # 'smallerReal'   # synthetic, TUS
# DATABASE = 'kglids_smaller_real_fine_grained_05'   # synthetic
FINE_GRAINED_DB = 'http://localhost:7200/repositories/tus_fine_grained_09' # 'tus_fine_grained_05' # 'smaller_real_fine_grained_05'
COARSE_GRAINED_DB = 'http://localhost:7200/repositories/tus_coarse_grained_09' #'tus_coarse_grained_05' #'smaller_real_coarse_grained_05'   # synthetic
NO_SUB_SAMPLING_DB = 'http://mossad-xps:7200/repositories/tus_fine_grained_no_sub_sampling_05' #'tus_fine_grained_no_sub_sampling_05'
# ************************************************
EXPERIMENT_NAME = 'precision_recall'
SAVE_RESULT_AS = EXPERIMENT_NAME + '_' + DATASET + '_' + 'fine_vs_coarse_grained' + '_' + str(THRESHOLD)
# ************************************************


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


def run_experiment(df, test_mapping: list, graph_handlers, exp_names, modes):
    query_tables = get_n_random_tables(df, NO_RANDOM_QUERY_TABLES)
    
    print('\n• running experiment!')
    top_k = []
    if DATASET == 'smallerReal':
        top_k = [5, 20, 35, 50, 65, 80, 95, 110, 125, 140, 155, 170, 185]
    elif DATASET == 'synthetic':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350]
    elif DATASET == 'TUS':
        top_k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    all_results = {}
    for graph, exp_name, mode in zip(graph_handlers, exp_names, modes):

        precisions = {}
        recalls = {}
        for table in tqdm.tqdm(query_tables):
            top_related_tables = get_top_k_related_tables(graph, table, max(top_k), THRESHOLD, mode=mode)
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
        print('='*50, exp_name, '='*50)
        print(exp_name, ': Average Precisions:\n', precisions_df.mean())
        print(exp_name, ': Average Recalls:\n', recalls_df.mean())
        
        
        res = {k: {'precision': np.mean(precisions_df[k]), 'recall': np.mean(recalls_df[k])} for k in top_k}
        all_results[exp_name] = res
        
    return all_results


def visualize(exp_results_dict):
    def plot_scores(top_k: list, exp_scores, exp_labels, colors, linestyles, markers, metric_name: str, title, 
                    show_legend=True, ymin=None, ymax=None):
        label_size = 14
        default_ticks = range(len(top_k))
        for exp_score, exp_label, color, linestyle, marker in zip(exp_scores, exp_labels, colors, linestyles, markers):
            plt.plot(default_ticks, exp_score, color, label=exp_label, marker=marker, linestyle=linestyle)
        plt.xticks(default_ticks, top_k)
        plt.ylim(ymin=ymin or 0, ymax= ymax or 1.01)
        plt.yticks(np.arange(ymin or 0, ymax or 1.1, 0.1))
        plt.xlabel('K', fontsize=label_size)
        plt.ylabel(metric_name, fontsize=label_size)
        # plt.title(title, y=-0.20, fontsize=label_size)
        plt.grid()
        if show_legend:
            plt.legend(loc='lower right')
        return plt

    exp_names = list(exp_results_dict)
    k = list(exp_results_dict[exp_names[0]])
    precisions = []
    recalls = []

    for exp_name in exp_names:
        exp_precision = []
        exp_recall = []
        for key in exp_results_dict[exp_name].keys():
            exp_precision.append(exp_results_dict[exp_name][key]['precision'])
            exp_recall.append(exp_results_dict[exp_name][key]['recall'])
        precisions.append(exp_precision)
        recalls.append(exp_recall)


    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    colors = ['forestgreen', 'black', 'gray', 'black']
    linestyles = ['solid', 'solid', 'dotted', 'dashed']
    markers = ['s', 'x', 'o', 'd']
    _ = plot_scores(k, precisions, exp_names, colors, linestyles, markers, 'Precision', '', ymin=0.4)
    plt.subplot(1, 2, 2)
    _ = plot_scores(k, recalls, exp_names, colors, linestyles, markers, 'Recall', '', show_legend=False, ymax=0.55)
    plt.tight_layout()
    plt.savefig('../plots/{}.pdf'.format(SAVE_RESULT_AS), dpi=300)
    plt.show()


def main():
    
    ground_truth_df = load_groundtruth(DATASET)
    ground_truth_set = get_table_mapping(ground_truth_df)
    
    t1 = time.time()
    fine_grained_graph = connect_to_graphdb(FINE_GRAINED_DB)
    coarse_grained_graph = connect_to_graphdb(COARSE_GRAINED_DB)
    no_sub_sampling_graph = connect_to_graphdb(NO_SUB_SAMPLING_DB)
    results_dict = run_experiment(ground_truth_df, ground_truth_set, 
                                  [fine_grained_graph, no_sub_sampling_graph, fine_grained_graph, coarse_grained_graph],
                                  ['KGLiDS', 'Fine-Grained (No Subsampling)', 'Fine-Grained', 'Coarse-Grained'],
                                  ['both', 'content', 'content', 'content'])
    print('Total time taken: ', time.time()-t1)

    visualize(results_dict)
    print('done.')


main()
