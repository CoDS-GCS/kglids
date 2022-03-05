import pandas as pd
import sys
import os
import random
import numpy as np
import tqdm
import pickle
from matplotlib import pyplot as plt
from helper.config import *
from helper.queries_7_hops import *

# *************EXPERIMENT PARAMETERS***********
THRESHOLD = 0.75
DATASET = 'smallerReal'
# ************CONFIGURATION PARAMETERS*********
EXPERIMENT_NAME = 'attribute_precision'
NAMESPACE = 'd3l_smallerReal'
SAVE_RESULT_AS = '060_' + str(EXPERIMENT_NAME) + "_" + str(THRESHOLD) + '_5_hops_' + DATASET
SPARQL = connect_to_blazegraph(NAMESPACE)
# *********************************************

def load_cache(load_as='cache'):
    with open('../cache/' + load_as + '.pkl', 'rb') as handle:
        return pickle.load(handle)


def load_groundtruth():
    file = df = 'null'
    if DATASET == 'smallerReal':
        file = '../gt_files/attr_gt.csv'
        print('loading {}'.format(file))
        df = pd.read_csv(file)
        df[df.columns[0]] = df[df.columns[0]] + '.csv'
        df[df.columns[2]] = df[df.columns[2]] + '.csv'
    elif DATASET == 'synthetic':
        # file = 'alignment_groundtruth.csv'
        # print('loading {}'.format(file))
        # df = pd.read_csv(file)
        print("not supported")
        sys.exit()
    return df


def get_n_random_tables(df: pd.DataFrame, n: int):
    print("getting {} random tables".format(n))
    query_tables = list(np.unique(df[df.columns[0]].to_list()))
    # print("unique query tables in groundtruth: ", len(query_tables))
    return random.sample(query_tables, n)


def attribute_precision(df, sk: list, target_table: str):
    def calculate_score(pred: list, test: list, join=[]):
        tp = fp = 0

        for pair in pred:
            # without J
            if len(join) == 0:
                if pair in test:
                    tp = tp + 1  # have it ground truth and have it in predictions
                else:
                    fp = fp + 1  # do not have it ground truth but have it in predictions
            # J
            else:
                if pair in test:
                    tp = tp + 1  # have it ground truth and have it in predictions
                else:
                    flag = False
                    for j in join:
                        if j in test:
                            flag = True
                            break
                    if flag == True:
                        tp = tp + 1
                    else:
                        fp = fp + 1

        precision = tp / (tp + fp)

        return precision

    # def calculate_score_j(pred: list, pred_j: list, test: list):
    #     tp = fp = 0

    #     for pair in pred:
    #         if pair in test:
    #             tp = tp + 1  # have it ground truth and have it in predictions
    #         else:
    #             fp = fp + 1  # do not have it ground truth but have it in predictions

    #     precision = tp / (tp + fp)

    #     return precision

    t_df = df.loc[df[df.columns[0]] == target_table]
    # t_df = t_df.drop(df.columns[0], 1)
    # t_df = t_df.drop(df.columns[2], 1)
    test = t_df.values.tolist()
    # print("test", len(test))
    # print(test[:5])

    precision = []
    precision_j = []
    for si in sk:
        # print(si[1])
        # print(target_table)
        # sys.exit()
        relatedness = get_related_columns_between_2_tables_attribute_precision(SPARQL, target_table, si[1], THRESHOLD)
        # print("relatedness: ", len(relatedness))
        # print(relatedness[:5])
        relatedness_j = get_related_columns_between_2_tables_j_attribute_precision(SPARQL, target_table, si[1],
                                                                                   THRESHOLD)
        # print("relatedness + j", len(relatedness_j))
        # print(relatedness_j[:5])
        # sys.exit()
        # pred_si = [relatedness[0], si[1], relatedness[1]]
        # print(pred_si)
        p = calculate_score(relatedness, test)
        if p == 1.0:
            p_j = p
        else:
            p_j = calculate_score(relatedness, test, relatedness_j)
        # print("Attribute precision: ",  p)
        # print("Attribute precision + J: ", p_j)
        # sys.exit()
        precision.append(p)
        precision_j.append(p_j)
        # pred_j[relatedness_j[0]]  = {}

    print("Attribute precision: ", np.mean(precision))
    print("Attribute precision + J: ", np.mean(precision_j))

    return np.mean(precision), np.mean(precision_j)


def run_experiment(df):
    def cache_score(dumping_file: dict, k: int, top_k: list):
        save_as = SAVE_RESULT_AS + '_k-{}'.format(k)
        with open('../cache/' + save_as + '.pkl', 'wb') as handle:
            pickle.dump(dumping_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('cached {}.pkl successfully!'.format(save_as))
        if k != top_k[0]:
            previous_k_value = top_k[top_k.index(k) - 1]
            rm_obj = '../cache/' + SAVE_RESULT_AS + '_k-{}'.format(previous_k_value) + '.pkl'
            if os.path.exists(rm_obj):
                os.remove(rm_obj)

    if os.path.exists("../cache/" + SAVE_RESULT_AS + ".txt"):
        os.remove("../cache/" + SAVE_RESULT_AS + ".txt")

    top_k = []

    if DATASET == 'smallerReal':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260]

    random_100_tables = get_n_random_tables(df, 100)
    res = {}
    print("running attribute precision experiment!")
    for k in top_k:
        ap_per_k = []
        ap_j_per_k = []
        print("K: ", k)
        for table in tqdm.tqdm(random_100_tables):
            sk = get_similar_relation_tables(SPARQL, table, k, 'semanticSimilarity', THRESHOLD)
            if len(sk):
                ap, ap_j = attribute_precision(df, sk, table)
                ap_per_k.append(ap)
                ap_j_per_k.append(ap_j)

        print(
            "Attribute precision for k: {} = {}\nAttribute precision +J for k: {} = {}".format(k, np.mean(ap_per_k), k,
                                                                                               np.mean(ap_j_per_k)))
        f = open("../cache/" + SAVE_RESULT_AS + ".txt", "a")
        f.write("K:{}\n\tattribute precision: {}\n\tattribute precision +J:{}\n\n".format(k, np.mean(ap_per_k),
                                                                                          np.mean(ap_j_per_k)))
        f.close()
        res[k] = {"attribute precision": np.mean(ap_per_k), "attribute precision + J": np.mean(ap_j_per_k)}
        cache_score(res, k, top_k)


def visualize(exp_res: dict):
    def plot_scores(k: list, metric: list, j: list, metric_name: str, d3l, aurum, d3l_j, aurum_j, tus):
        default_ticks = range(len(k))
        plt.plot(default_ticks, metric, 'g', label='KGLiDS', marker="x")
        plt.plot(default_ticks, j, 'limegreen', label='KGLiDS+Join', marker="x")
        plt.plot(default_ticks, d3l, 'cornflowerblue', label='D3L', marker="s")
        plt.plot(default_ticks, d3l_j, 'blue', label='D3L+Join', marker="s")
        plt.plot(default_ticks, aurum, 'darkorange', label='Aurum', marker="d")
        plt.plot(default_ticks, aurum_j, 'red', label='Aurum+Join', marker="d")
        plt.plot(default_ticks, tus, 'gray', label='TUS', marker="^")
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

    scores_ap = pd.read_csv('../d3l_scores/attribute_precision_smallerReal.csv')
    d3l = scores_ap['D3L']
    d3l_j = scores_ap['D3L+J']
    aurum = scores_ap['Aurum']
    aurum_j = scores_ap['Aurum+J']
    tus = scores_ap['TUS']

    k = []
    ap = []
    ap_j = []
    for key, v in exp_res.items():
        k.append(key)
        ap.append(v['attribute precision'])
        ap_j.append(v['attribute precision + J'])

    # scores_ap['KGLids'] = ap
    # scores_ap['KGLids+J'] = ap_j
    # scores_ap.to_csv('d3l_scores/attribute_precision_smallerReal_kglids.csv')

    # plt.figure(figsize=(13, 5))
    # plt.subplot(1, 2, 1)
    # fig1 = plot_scores(k, precision, '(a) Precision', d3l_precision, aurum_precision, tus_precision)
    # plt.subplot(1, 2, 2)
    # fig2 = plot_scores(k, recall, '(b) Recall', d3l_recall, aurum_recall, tus_recall)
    # plt.tight_layout()
    # ap.extend([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    # print(ap)
    plt.figure(figsize=(7.5, 5))

    plot_scores(k, ap, ap_j, 'Attribute precision', d3l, aurum, d3l_j, aurum_j, tus)
    #plt.savefig('d3l_scores/attribute_precision_smallerReal.pdf')
    plt.show()


def main():
    #df = load_groundtruth()
    # print(df.head())

    # print(df.loc[df[df.columns[0]] == '35To5x3.csv'])

    #run_experiment(df)
    exp_res = load_cache('{}_k-260'.format(SAVE_RESULT_AS))
    # print(exp_res)
    visualize(exp_res)


main()
