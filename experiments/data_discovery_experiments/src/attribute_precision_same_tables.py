import argparse
import dask.dataframe as dd
import random
import tqdm
import time
import multiprocessing as mp

from helper.config import *
from helper.queries import *
from helper.cache import *
from helper.plot import *
from helper.comparsion_plot import plot_comparison

# **************************CONFIGURATIONS*****************************
THRESHOLD = 0.75
EXPERIMENT_NAME = 'attribute_precision'
DATASET = 'smallerReal'   # synthetic
DATABASE = 'kglids_smaller_real_fine_grained_09'   # synthetic
MAX_K = 260
RESULT_SAVE_PATH = f'../cache/attribute_precision_smallerReal_k-{MAX_K}.pkl'
# *********************************************************************
SAVE_RESULT_AS = EXPERIMENT_NAME + '_' + DATASET + '_' + str(THRESHOLD)

SPARQL = None
# *********************************************************************


def load_cache(load_as='cache'):
    with open(load_as, 'rb') as handle:
        return pickle.load(handle)


def load_ground_truth():
    print('Loading ground-truth', end=' ')
    if DATASET == 'smallerReal':
        file = '../gt_files/attr_gt.csv'
        print(file, end=' ')
        df = dd.read_csv(file)
        df[df.columns[0]] = df[df.columns[0]] + '.csv'
        df[df.columns[2]] = df[df.columns[2]] + '.csv'
    else:
        file = '../gt_files/att_groundtruth.csv'
        print(file, end=' ')
        df = dd.read_csv(file)
    print('\tdone.')
    return df


def get_n_random_tables(df, n: int):
    print('Getting {} random tables '.format(n), end='')
    t1 = time.time()
    random_samples = random.sample(list(df[df.columns[0]].unique().compute()), n)
    print('\tdone, time taken: {}.'.format(time.time() - t1))
    return random_samples


def attribute_precision(ground_truth: set, query_table: str, k_related_tables: list):
    def calculate_attribute_precision_without_join(
            predicted_pairs: list, ground_truth_pairs: set):
        tp = fp = 0
        for pair in predicted_pairs:
            if pair in ground_truth_pairs:
                tp = tp + 1
            else:
                fp = fp + 1

        return tp / (tp + fp)

    def calculate_attribute_precision_with_join(
            predicted_pairs: list, join_paths: list, ground_truth_pairs: set):
        tp = fp = 0
        for pair in predicted_pairs:
            if pair in ground_truth_pairs:
                tp = tp + 1
            else:
                flag = False
                for j in join_paths:
                    if j[1] == pair[1] and j in ground_truth_pairs:
                        flag = True
                        break
                if flag:
                    tp = tp + 1
                else:
                    fp = fp + 1

        return tp / (tp + fp)

    precision = []
    precision_j = []

    for table in k_related_tables:
        pred_without_joins = get_related_columns_between_2_tables_attribute_precision(
            SPARQL, query_table, table[1], THRESHOLD)

        attr_precision = calculate_attribute_precision_without_join(pred_without_joins, ground_truth)

        if attr_precision == 1.0:
            attr_precision_with_join = attr_precision
            precision_j.append(attr_precision_with_join)
        else:
            pred_with_joins = get_related_columns_between_2_tables_j_attribute_precision(
                SPARQL, query_table, table[1], THRESHOLD)
            attr_precision_with_join = calculate_attribute_precision_with_join(pred_without_joins, pred_with_joins,
                                                                               ground_truth)
        precision.append(attr_precision)
        precision_j.append(attr_precision_with_join)

    return np.mean(precision), np.mean(precision_j)


def run_experiment(df):
    ground_truth_per_query_table = {}

    def get_ground_truth_for_query_table(table: str):
        if table in ground_truth_per_query_table:
            return ground_truth_per_query_table.get(table)
        else:
            gd = df.loc[df[df.columns[0]] == table].compute()
            gd = set([tuple(x) for x in gd.values])
            ground_truth_per_query_table[table] = gd
            return ground_truth_per_query_table.get(table)

    if os.path.exists('../cache/' + SAVE_RESULT_AS + '.txt'):
        os.remove('../cache/' + SAVE_RESULT_AS + '.txt')

    random_100_tables = get_n_random_tables(df, 100)
    top_k = []
    if DATASET == 'smallerReal':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260]
    elif DATASET == 'synthetic':
        top_k = [5, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350]
        df = df[['query_table', 'query_col_name', 'candidate_table', 'candidate_col_name']]

    print("\nRunning '{}' experiment on '{}' dataset. Top-k values = {}".
          format(EXPERIMENT_NAME.replace('_', ' ').upper(), DATASET.upper(), top_k))

    res = {}
    for k in top_k:
        print('\nComputing for K =', k)
        ap_per_k = []
        ap_j_per_k = []
        for query_table in tqdm.tqdm(random_100_tables):  # average over 100 random tables
            k_related_tables = get_top_k_related_tables(SPARQL, query_table, k, THRESHOLD)
            ground_truth = get_ground_truth_for_query_table(query_table)

            if len(k_related_tables):
                ap, ap_j = attribute_precision(ground_truth, query_table, k_related_tables)
                ap_per_k.append(ap)
                ap_j_per_k.append(ap_j)
                """
                print('\n• Mean Attribute precision  : ', np.mean(ap_per_k))
                print('• Mean Attribute precision + Join : ', np.mean(ap_j_per_k))
                """
            else:
                print('bad table ', query_table)

        print("Attribute precision for k: {} = {}\n"
              "Attribute precision +J for k: {} = {}".format(k, np.mean(ap_per_k), k, np.mean(ap_j_per_k)))
        f = open("../cache/" + SAVE_RESULT_AS + ".txt", "a")
        f.write("K:{}\n\tattribute precision: {}\n\tattribute precision +J:{}\n\n".format(k, np.mean(ap_per_k),
                                                                                          np.mean(ap_j_per_k)))
        f.close()
        res[k] = {"attribute precision": np.mean(ap_per_k), "attribute precision + J": np.mean(ap_j_per_k)}

        cache_score(res, k, top_k, SAVE_RESULT_AS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=RESULT_SAVE_PATH)
    parser.add_argument('--graph-name', type=str, default=DATABASE)
    args = parser.parse_args()
    global SPARQL
    SPARQL = connect_to_stardog(db=args.graph_name)
    df = load_ground_truth()
    t1 = time.time()
    run_experiment(df)
    print('\nTotal time taken: ', time.time() - t1)

    exp_res = load_cache(args.save_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    attribute_precision_plot = visualize(exp_res, EXPERIMENT_NAME.replace('_', ' ').capitalize(), DATASET)
    plt.subplot(1, 2, 2)
    comparison_plot = plot_comparison()
    plt.tight_layout()
    plt.savefig('../plots/{}.pdf'.format(EXPERIMENT_NAME), dpi=300)
    print('done.')


main()
