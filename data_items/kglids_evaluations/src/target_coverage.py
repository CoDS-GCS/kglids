from curses.panel import top_panel
from distutils.command.install_egg_info import to_filename
import sys
import tqdm
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON
from helper.queries_7_hops import *
from helper.config import *

# *************EXPERIMENT PARAMETERS***********
THRESHOLD = 0.75
DATASET = 'smallerReal'
# ************CONFIGURATION PARAMETERS*********
EXPERIMENT_NAME = 'target_coverage'
ANAYLYSIS_FILE = DATASET + EXPERIMENT_NAME + str(THRESHOLD)
NAMESPACE = 'd3l_smallerReal'
SAVE_RESULT_AS = '050_' +EXPERIMENT_NAME + "_" + str(THRESHOLD) + '_7_hops_' + DATASET
#SPARQL = connect_to_blazegraph(NAMESPACE)
# *********************************************

if os.path.exists('analysis/{}.txt'.format(ANAYLYSIS_FILE)):
    os.remove('analysis/{}.txt'.format(ANAYLYSIS_FILE))

def load_cache(load_as='../cache'):
    with open('../cache/' + load_as + '.pkl', 'rb') as handle:
        return pickle.load(handle)

counter = 1
processed_k = []
def calculate_coverage(k, SPARQL, target_table: str, sk: list, arity_target_table):
    
    def put_to_analyzer(k, target, candidate, arity_related, arity, arity_target):
        global counter
        
        if counter <=2 or k not in processed_k:
            print("updating ",'analysis/{}.txt'.format(ANAYLYSIS_FILE))
            info = "{}. K = {}\ntarget table: {}\n\t- candidate table: {}\n\t- target coverage: {}/{}={}\n\t- target coverage + Join: {}/{}={}\n**************************************************\n".format(counter,k,target, candidate, arity, arity_target, (arity/arity_target), arity_related, arity_target, (arity_related/arity_target))
            dest = 'analysis/{}.txt'.format(ANAYLYSIS_FILE)
            with open(dest, "a") as f:
                f.write(info)
            counter = counter + 1
            processed_k.append(k)
    
    #print("calculating coverage for: {} and {}".format(target_table, sk))
    coverage_t = []
    coverage_t_j = []
    
   
    for si in sk:
        
        
        relatedness = get_related_columns_between_2_tables(SPARQL, target_table, si[1], THRESHOLD)
        #print("related tables: ", len(set(relatedness)))
        coverage = (len(set(relatedness)))/(arity_target_table)    
        #print("{}/{}={}".format(len(set(relatedness)), arity_target_table, coverage))
        #print("coverage = " ,coverage)
        coverage_t.append(coverage) 
        
        if coverage != 1.0:
            relatedness_j = get_related_columns_between_2_tables_j(SPARQL, target_table, si[1], THRESHOLD)
            relatedness_j = relatedness + relatedness_j
            coverage_j = (len(set(relatedness_j)))/(arity_target_table)
        #     jsi = get_joinable_table_paths(SPARQL, si[1])
        #     relatedness_j = []
        #     for jl in jsi:
        #         relatedness_j.extend(get_related_columns_between_2_tables_j(SPARQL, target_table, jl))
        #         if (len(set(relatedness_j))) == arity_target_table:
        #             break
        #     coverage_j = (len(set(relatedness_j)))/(arity_target_table)  
        #     coverage_t_j.append(coverage_j) 
        else:
            coverage_j = 1.0
        coverage_t_j.append(coverage_j)   
        # if coverage_j <= 0.5:
        #     put_to_analyzer(k, target_table, si[1], (len(set(relatedness_j))), (len(set(relatedness))), (arity_target_table))
        
    return np.mean(coverage_t), np.mean(coverage_t_j)


def visualize(exp_res: dict):
    def plot_scores(k: list, metric: list, j: list, metric_name: str, d3l, aurum, tus, d3l_j, aurum_j):
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

    scores_coverage = pd.read_csv('../d3l_scores/coverage_smallerReal.csv')
    d3l = scores_coverage['D3L']
    d3l_j = scores_coverage['D3L+J']
    aurum = scores_coverage['Aurum']
    aurum_j = scores_coverage['Aurum+J']
    tus = scores_coverage['TUS']


    k = []
    coverage = []
    coverage_j = []
    for key, v in exp_res.items():
        k.append(key)
        coverage.append(v['coverage'])
        coverage_j.append(v['coverage+J'])



    #scores_coverage['KGLids'] = coverage
    #scores_coverage['KGLids+J'] = coverage_j
    #scores_coverage.to_csv('d3l_scores/coverage_smallerReal_kglids.csv')

    # plt.figure(figsize=(13, 5))
    # plt.subplot(1, 2, 1)
    # fig1 = plot_scores(k, precision, '(a) Precision', d3l_precision, aurum_precision, tus_precision)
    # plt.subplot(1, 2, 2)
    # fig2 = plot_scores(k, recall, '(b) Recall', d3l_recall, aurum_recall, tus_recall)
    # plt.tight_layout()
    plt.figure(figsize=(7.5, 5))
    plot_scores(k, coverage, coverage_j, 'Target coverage', d3l, aurum, tus, d3l_j, aurum_j)
    #plt.savefig('d3l_scores/exp10.pdf')
    plt.show()


def run_experiment():
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
            
    if os.path.exists("../cache/"+SAVE_RESULT_AS+".txt"):
        os.remove("../cache/"+SAVE_RESULT_AS+".txt")
    
    print("running target coverage experiment!")
    top_k = []

    if DATASET == 'smallerReal':
        top_k = [1,2, 3, 5, 20, 50, 80, 110, 140, 170, 200, 230, 260]
    
    random_100_tables = random.sample(get_all_profiled_tables(SPARQL), 100)
    res = {}
    for k in top_k:
        coverage_per_k = []
        coverage_j_per_k = []
        for table in tqdm.tqdm(random_100_tables): # avg. over 100 tables
            # sk1 = get_similar_relation_tables(SPARQL, table, k, 'contentSimilarity')
            # sk2 = get_similar_relation_tables(SPARQL, table, k, 'inclusionDependency')
            # sk = sk1+sk2
            # sk.sort()
            # sk = list(sk for sk,_ in itertools.groupby(sk))
            sk = get_similar_relation_tables(SPARQL, table, k, 'semanticSimilarity', THRESHOLD)
            # top_k = Sk i.e. k-most related datasets for given Target 'T'

            if len(sk):
                arity_table = len(get_all_columns(SPARQL, table))
                # print("arity for table {} = {}".format(table, arity_table))
                #coverage = calculate_coverage(k, SPARQL, table, sk, arity_table)
                coverage, coverage_j = calculate_coverage(k, SPARQL, table, sk, arity_table)
                coverage_per_k.append(coverage)
                coverage_j_per_k.append(coverage_j)
                print("K: ", k)
                print("coverage:", coverage)
                print("J: ", coverage_j)
        print("coverage for k: {} = {}".format(k, np.mean(coverage_per_k)))
        print("coverage+J for k: {} = {}".format(k, np.mean(coverage_j_per_k)))
        f = open("../cache/"+SAVE_RESULT_AS+".txt", "a")
        f.write("K:{}\n\tcoverage:{}\n\tcoverage+J:{}\n\n".format(k, np.mean(coverage_per_k), np.mean(coverage_j_per_k)))
        f.close()
        res[k] = {"coverage" : np.mean(coverage_per_k), "coverage+J" : np.mean(coverage_j_per_k)}
        cache_score(res, k, top_k)

def main():
    #start = time.time()
    #run_experiment()
    #print("\ntime taken: ", time.time()-start)
    exp_res = load_cache('{}_k-260'.format(SAVE_RESULT_AS))
    visualize(exp_res)

main()