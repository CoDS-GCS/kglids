import tqdm
import time
import random
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

from helper.queries import *
from helper.config import connect_to_stardog

# *************EXPERIMENT PARAMETERS**************
THRESHOLD = 1.0
NO_RANDOM_QUERY_TABLES = 100
DATASET = 'santos'
DATABASE = 'kglids_santos_fine_grained_075'
MAX_K = 10
# ************************************************
EXPERIMENT_NAME = 'santos_precision_recall'
SAVE_RESULT_AS = EXPERIMENT_NAME + '_' + DATASET
SPARQL = connect_to_stardog(db=DATABASE)
# ************************************************


def cache_score(dumping_file: dict, k: int, top_k: list, save_as: str):
    with open('../../cache/' + save_as + '_k-{}'.format(k)+ '.pkl', 'wb') as handle:
        pickle.dump(dumping_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('cached {}.pkl successfully!'.format(save_as + '_k-{}'.format(k)))
    if k != top_k[0]:
        previous_k_value = k - 1
        rm_obj = '../cache/' + save_as + '_k-{}'.format(previous_k_value) + '.pkl'
        if os.path.exists(rm_obj):
            os.remove(rm_obj)


def load_cache(load_as='cache'):
    with open('../../cache/' + load_as, 'rb') as handle:
        return pickle.load(handle)



def get_table_mapping(df: pd.DataFrame):
    print("getting mapping between tables from ground truth")
    query_tables = df[df.columns[0]].to_list()
    candidate_tables = df[df.columns[1]].to_list()
    mapping = []
    for i in range(len(query_tables)):
        mapping.append([query_tables[i], candidate_tables[i]])

    return mapping


def get_n_random_tables(df: pd.DataFrame, n: int):
    print("getting {} random tables".format(n))
    query_tables = list(np.unique(df[df.columns[0]].to_list()))
    return random.sample(query_tables, n)


def calculate_scores(pred: list, test: list):
    tp = fp = 0

    for pair in pred:
        if pair in test:
            tp = tp + 1  # have it ground truth and have it in predictions
        else:
            fp = fp + 1  # do not have it ground truth but have it in predictions

    test2 = [i for i in test if i[0] == pred[0][0]]
    fn = len([i for i in test2 if i not in pred])

    precision = tp / (tp + fp) if tp != 0 else 0
    recall = tp / (tp + fn) if tp != 0 else 0

    return precision, recall


def run_experiment(df, test_mapping: list):
    with open('santos_query_tables.txt') as f:
        query_tables = [line.strip() for line in f.readlines()]
        
    print('\n• running experiment!')
    top_k = list(range(1, 11))

    res = {}
    for k in top_k:
        print("\ncalculating scores for k = {}".format(k))

        precision_per_table = []
        recall_per_table = []

        for table in tqdm.tqdm(query_tables):
            predicted_mapping = get_top_k_related_tables(SPARQL, table, k, THRESHOLD)
            precision, recall = calculate_scores(predicted_mapping, test_mapping)
            precision_per_table.append(precision)
            recall_per_table.append(recall)
        print("Avg. precision for k = {} : {}".format(k, np.mean(precision_per_table)))
        print("Avg. recall for k = {} : {}".format(k, np.mean(recall_per_table)))
        res[k] = {'precision': np.mean(precision_per_table), 'recall': np.mean(recall_per_table)}
        cache_score(res, k, top_k, SAVE_RESULT_AS)



def main():

    ground_truth_df = pd.read_csv('santos_ground_truth.csv')
    test_mapping = get_table_mapping(ground_truth_df)
    run_experiment(ground_truth_df, test_mapping)



main()
