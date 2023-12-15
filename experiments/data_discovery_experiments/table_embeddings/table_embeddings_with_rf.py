from datetime import datetime
import os
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


def generate_train_and_valid_sets(ground_truth, training_tables, table_embeddings):
    # generate positive and negative matches
    filtered_ground_truth = {i for i in ground_truth if i[0] in training_tables and i[1] in training_tables}
    # 1. positive matches is the ground truth filtered by training tables
    positive_matches = list(filtered_ground_truth)
    
    # 2. negative matches are random tables (and not present in positive matches)
    negative_matches = []
    for match in positive_matches:
        target_table = match[0]
        random_table = random.sample(training_tables, 1)[0]
        while (target_table, random_table) not in filtered_ground_truth:
            random_table = random.sample(training_tables, 1)[0]
        negative_matches.append((target_table, random_table))
    
    # we have now equal positive and negative matches.
    # convert from matches to training format: (embed1, embed2, 1/0)
    
    features = [np.concatenate([table_embeddings[match[0]], table_embeddings[match[1]]])
                         for match in positive_matches + negative_matches]
    targets = [1] * len(positive_matches) + [0] * len(negative_matches)
    
    features, targets = shuffle(features, targets)
    
    train_x, valid_x, train_y, valid_y = train_test_split(features, targets, test_size=0.2)
    
    return train_x, valid_x, train_y, valid_y
    

def calculate_unionability_scores(table, data_lake_tables, table_embeddings, model):
    features = [np.concatenate([table_embeddings[table], table_embeddings[table2]]) for table2 in data_lake_tables]
    scores = model.predict_proba(features)[:,1]
    return scores.tolist()


def calculate_precision_recall(query_table, table_embeddings, k_range, model, ground_truth):
    
    # 1. calculate unionability score between this table and all other tables
    data_lake_tables = [table for table in table_embeddings if table != query_table]
    unionability_scores = calculate_unionability_scores(query_table, data_lake_tables, table_embeddings, model)
    table_scores = list(zip(data_lake_tables, unionability_scores))
 
    top_tables = [i[0] for i in sorted(table_scores, key=lambda x: x[1], reverse=True) if i[1] >= 0.5] 
    
    # 2. calculate precision and recall at different values of k
    precisions = []
    recalls = []
    
    for k in k_range:
        predicted_tables = set(top_tables[:k]) 

        tp, fp = 0, 0

        if not predicted_tables:
            precisions.append(0)
            recalls.append(0)
            continue

        for table in predicted_tables:
            if (query_table, table) in ground_truth:
                tp = tp + 1  # have it ground truth and have it in predictions
            else:
                fp = fp + 1  # do not have it ground truth but have it in predictions

        all_unionable_tables = {pair[1] for pair in ground_truth if pair[0] == query_table}
        fn = len([table for table in all_unionable_tables if table not in predicted_tables])

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls


def visualize(table_embeddings_precisions, table_embeddings_recalls, k_range, baselines_precisions, baselines_recalls):
    def plot_scores(top_k: list, metric: list, metric_name: str, title, starmie, santos):
        label_size = 17
        default_ticks = range(len(top_k))
        plt.plot(default_ticks, starmie, 'orange', linestyle='--', label='Starmie', marker="x")
        plt.plot(default_ticks, santos, 'crimson', linestyle='--', label='SANTOS', marker="o")
        plt.plot(default_ticks, metric, 'forestgreen', label='RF w/ KGLiDS TE ', marker="s")
        plt.xticks(default_ticks, top_k)
        plt.ylim(ymin=0)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('K', fontsize=label_size)
        plt.ylabel(metric_name, fontsize=label_size)
        plt.title(title, y=-0.20, fontsize=label_size)
        plt.legend(loc='lower right')
        plt.grid()
        return plt
    
    santos_precisions = baselines_precisions['SANTOS']
    santos_recalls = baselines_recalls['SANTOS']
    starmie_precisions = baselines_precisions['Starmie']
    starmie_recalls = baselines_recalls['Starmie']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    _ = plot_scores(k_range, table_embeddings_precisions, 'Precision', '(a)', starmie_precisions, santos_precisions)
    plt.subplot(1, 2, 2)
    _ = plot_scores(k_range, table_embeddings_recalls, 'Recall', '(b)', starmie_recalls, santos_recalls)
    plt.tight_layout()
    plt.savefig('table_embeddings_p_r.pdf', dpi=300)
    plt.show()


def main():
    
    TABLE_EMBEDDINGS_PATH = os.path.expanduser('~/projects/kglids/storage/embeddings/smaller_real_table_embeddings.pickle')
    GROUND_TRUTH_PATH = os.path.expanduser('~/projects/kglids/experiments/data_discovery_experiments/gt_files/ds_gt.csv')
    D3L_PRECISIONS_PATH = os.path.expanduser('~/projects/kglids/experiments/data_discovery_experiments/d3l_scores/precision_smallerReal.csv')
    D3L_RECALLS_PATH = os.path.expanduser('~/projects/kglids/experiments/data_discovery_experiments/d3l_scores/recall_smallerReal.csv')
    K_RANGE = [5, 20, 35, 50, 65, 80, 95, 110, 125, 140, 155, 170, 185]
    
    table_embeddings = pd.read_pickle(TABLE_EMBEDDINGS_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    ground_truth['target_ds'] = ground_truth['target_ds'] + '.csv'
    ground_truth['candidate_ds'] = ground_truth['candidate_ds'] + '.csv'
    ground_truth = set(zip(ground_truth['target_ds'], ground_truth['candidate_ds']))
    
    test_tables = random.choices(list(table_embeddings), k=100)
    train_tables = {i for i in table_embeddings if i not in test_tables}
    
    print(datetime.now(), ': Generating training and valid sets.')
    train_x, valid_x, train_y, valid_y = generate_train_and_valid_sets(ground_truth, train_tables, table_embeddings)
    print(datetime.now(), 'Done. Train Samples:', len(train_x), '; Valid Samples:', len(valid_x))
    
    print(datetime.now(), 'Training RandomForestClassifier.')
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=123)
    model.fit(train_x, train_y)
    
    print(datetime.now(), 'Done. Predicting test samples.')
    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)
    train_f1, train_acc = f1_score(train_y, train_pred), accuracy_score(train_y, train_pred)
    valid_f1, valid_acc = f1_score(valid_y, valid_pred), accuracy_score(valid_y, valid_pred)
    print('Train: F1-Score:', train_f1, '; Accuracy:', train_acc)
    print('Valid: F1-Score:', valid_f1, '; Accuracy:', valid_acc)
    
    print(datetime.now(), 'Calculating precision/recall for', len(test_tables), 'query tables')
    all_table_precisions = []
    all_table_recalls = []
    for table in tqdm(test_tables):
        precisions, recalls = calculate_precision_recall(table, table_embeddings, K_RANGE, model, ground_truth)
        all_table_precisions.append(precisions)
        all_table_recalls.append(recalls)
        
    all_table_precisions = pd.DataFrame(all_table_precisions, columns=K_RANGE)
    average_precisions = all_table_precisions.mean()
    all_table_recalls = pd.DataFrame(all_table_recalls, columns=K_RANGE)
    average_recalls = all_table_recalls.mean()
    
    print(datetime.now(), 'Average precisions:', average_precisions)
    print(datetime.now(), 'Average Recalls:', average_recalls)
    
    d3l_precisions = pd.read_csv(D3L_PRECISIONS_PATH)
    d3l_recalls = pd.read_csv(D3L_RECALLS_PATH)
    visualize(average_precisions, average_recalls, K_RANGE, d3l_precisions, d3l_recalls)
    

if __name__ == '__main__':
    main()