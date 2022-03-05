import tqdm
import pickle
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON
from helper.queries import *

# *************EXPERIMENT PARAMETERS***********
NO_RANDOM_QUERY_TABLES = 100
DATASET = 'synthetic'
# ************CONFIGURATION PARAMETERS*********
NAMESPACE = 'd3l_synthetic_backup'
GROUND_TRUTH = 'alignment_groundtruth.csv'
SAVE_RESULT_AS = 'd3l_synthetic'
# *********************************************

def load_cache(load_as='cache'):
    with open('cache/' + load_as + '.pkl', 'rb') as handle:
        return pickle.load(handle)


def load_groundtruth():
    file = df = 'null'
    if DATASET == 'smallerReal':
        file = 'ds_gt.csv'
        print('loading {}'.format(file))
        df = pd.read_csv(file)
        df[df.columns[0]] = df[df.columns[0]] + '.csv'
        df[df.columns[1]] = df[df.columns[1]] + '.csv'
    elif DATASET == 'synthetic':
        file = 'alignment_groundtruth.csv'
        print('loading {}'.format(file))
        df = pd.read_csv(file)
    return df

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
    # print("unique query tables in groundtruth: ", len(query_tables))
    return random.sample(query_tables, n)


def connect_to_blazegraph():
    endpoint = 'http://localhost:19999/blazegraph/namespace/' + NAMESPACE + '/sparql'
    print('connected to blazegraph ({})'.format(endpoint))
    return SPARQLWrapper(endpoint)


def execute_query(sparql, query: str):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def table_profiled(sparql, table_name: str):
    res = execute_query(sparql, check_if_table_exists(table_name))
    if res['boolean']:
        return True


def get_profiled_tables(sparql):
    print("getting all profiled tables")
    result = []
    res = execute_query(sparql, get_all_profiled_tables())
    for r in res["results"]["bindings"]:
        result.append(r["table_name"]["value"])
    return result


def get_top_k_pairs(sparql, table_name: str, k: int):
    result = []
    res = execute_query(sparql, get_top_k(table_name, k))
    for r in res["results"]["bindings"]:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        result.append([table1, table2])

    return result


def get_semantic_similar_tables(sparql, query_table: str, k: int):
    result = []
    res = execute_query(sparql, get_semantic_similar_tables_query(query_table))
    for r in res["results"]["bindings"]:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        certainty = r["certainty"]["value"]
        result.append([table1, table2, certainty])
    result = get_top_k_query(result, k)
    return result


def calculate_scores(pred: list, test: list, query_tables: list):
    tp = fp = fn = 0

    for pair in pred:
        if pair in test:
            tp = tp + 1  # have it ground truth and have it in predictions
        else:
            fp = fp + 1  # do not have it ground truth but have it in predictions

    test2 = [i for i in test if i[0] == pred[0][0]]
    fn = len([i for i in test2 if i not in pred])
    # for pair in test2:
    #     if  pair not in pred:
    #         fn = fn + 1  # have it ground truth but not in predictions

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def run_experiment(sparql, test_mapping: list):
    def cache_score(dumping_file: dict, k: int):
        save_as = SAVE_RESULT_AS + '_k-{}'.format(k)
        # if save_as == SAVE_RESULT_AS + '_k-185':
        with open('cache/' + save_as + '.pkl', 'wb') as handle:
            pickle.dump(dumping_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('cached {}.pkl successfully!'.format(save_as))

    query_tables = get_n_random_tables(load_groundtruth(), NO_RANDOM_QUERY_TABLES)
    print('\n• running experiment!')
    top_k = []
    if DATASET == 'smallerReal':
        top_k = [5, 20, 35, 50, 65, 80, 95, 110, 125, 140, 155, 170, 185]
    elif DATASET == 'synthetic':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350]
    res = {}
    for k in top_k:
        print("\ncalculating scores for k = {}".format(k))

        precision_per_table = []
        recall_per_table = []

        for table in tqdm.tqdm(query_tables):
            predicted_mapping = get_semantic_similar_tables(sparql, table, k)
            precision, recall = calculate_scores(predicted_mapping, test_mapping, query_tables)
            precision_per_table.append(precision)
            recall_per_table.append(recall)
        print("Avg. precision for k = {} : {}".format(k, np.mean(precision_per_table)))
        print("Avg. recall for k = {} : {}".format(k, np.mean(recall_per_table)))
        res[k] = {'precision': np.mean(precision_per_table), 'recall': np.mean(recall_per_table)}
        cache_score(res, k)


def visualize(exp_res: dict):
    def plot_scores(k: list, metric: list, metric_name: str, d3l, aurum, tus):
        default_ticks = range(len(k))
        plt.plot(default_ticks, metric, 'g', label='KGLiDS', marker="x")
        plt.plot(default_ticks, d3l, 'cornflowerblue', label='D3L', marker="s")
        plt.plot(default_ticks, aurum, 'darkorange', label='Aurum', marker="d")
        plt.plot(default_ticks, tus, 'gray', label='TUS', marker="^")
        plt.xticks(default_ticks, k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K')
        plt.ylabel(metric_name.strip('(ab)'))
        plt.title(metric_name)
        plt.legend(loc='lower right')
        plt.grid()
        return plt

    scores_precision = pd.read_csv('d3l_scores/precision.csv')
    scores_recall = pd.read_csv('d3l_scores/recall.csv')
    d3l_precision = scores_precision['D3L']
    d3l_recall = scores_recall['D3L']
    aurum_precision = scores_precision['Aurum']
    aurum_recall = scores_recall['Aurum']
    tus_precision = scores_precision['TUS']
    tus_recall = scores_recall['TUS']

    k = []
    precision = []
    recall = []
    for key, v in exp_res.items():
        k.append(key)
        precision.append(v['precision'])
        recall.append(v['recall'])

    scores_precision['KGLids'] = precision
    scores_recall['KGLids'] = recall
    scores_precision.to_csv('d3l_scores/precision_kglids.csv')
    scores_recall.to_csv('d3l_scores/recall_kglids.csv')

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    fig1 = plot_scores(k, precision, '(a) Precision', d3l_precision, aurum_precision, tus_precision)
    plt.subplot(1, 2, 2)
    fig2 = plot_scores(k, recall, '(b) Recall', d3l_recall, aurum_recall, tus_recall)
    plt.tight_layout()
    plt.savefig('d3l_scores/exp4.pdf')
    #plt.show()



def main():
    df = load_groundtruth()
    test_mapping = get_table_mapping(df)
    sparql = connect_to_blazegraph()
    # profiled_tables = get_profiled_tables(sparql)
    run_experiment(sparql, test_mapping)
    #exp_res = load_cache('d3l_smallerReal_k-185')
    #visualize(exp_res)


main()
