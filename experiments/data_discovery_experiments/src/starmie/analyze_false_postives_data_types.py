import os.path

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from glob import glob
import json
from tqdm import tqdm

def main():
    
    smaller_real_profiles_path = os.path.expanduser('~/projects/kglids/storage/profiles/smaller_real_profiles')
    smaller_real_ground_truth_path = os.path.expanduser('~/projects/kglids/storage/data_sources/dataset_storage/experiments/smallerReal/groundtruth/attr_gt.csv')
    starmie_matches_path = os.path.expanduser('~/projects/SiMa/ground_truth/sima_smaller_real_starmie_groundtruth.csv')
    
    ground_truth = pd.read_csv(smaller_real_ground_truth_path)
    ground_truth = {tuple(i) for i in ground_truth.values}
    starmie_matches = pd.read_csv(starmie_matches_path)
    starmie_matches = starmie_matches[starmie_matches.columns.tolist()[:4]]
    starmie_matches = {tuple(i) for i in starmie_matches.values}
    
    starmie_true_positives = [i for i in starmie_matches if i in ground_truth]
    starmie_false_postives = [i for i in starmie_matches if i not in ground_truth]
    starmie_false_negatives = [i for i in ground_truth if i not in starmie_matches]
    
    
    # get data types for all columns
    files = glob(os.path.join(smaller_real_profiles_path, '**/*.json'))
    column_types = {}
    for file in tqdm(files):
        with open(file, 'r') as f:
            profile = json.load(f)
        column_types[(profile['table_name'].replace('.csv', ''), profile['column_name'])] = profile['data_type']
    
    numerical_columns = {i for i in column_types if column_types[i] in ['boolean', 'float', 'int']}
    numerical_matches = [i for i in starmie_matches if
                         (i[0], i[1]) in numerical_columns or (i[2], i[3]) in numerical_columns]
    non_numerical_matches = [i for i in starmie_matches if
                             (i[0], i[1]) not in numerical_columns and (i[2], i[3]) not in numerical_columns]
    numerical_true_postivies = [i for i in starmie_true_positives if
                                 (i[0], i[1]) in numerical_columns or (i[2], i[3]) in numerical_columns]
    non_numerical_true_postivies = [i for i in starmie_true_positives if
                                    (i[0], i[1]) not in numerical_columns and (i[2], i[3]) not in numerical_columns]
    numerical_false_postivies = [i for i in starmie_false_postives if
                                 (i[0], i[1]) in numerical_columns or (i[2], i[3]) in numerical_columns]
    numerical_false_negatives = [i for i in starmie_false_negatives if
                                 (i[0], i[1]) in numerical_columns or (i[2], i[3]) in numerical_columns]
    
    numerical_ground_truth = [i for i in ground_truth if
                                 (i[0], i[1]) in numerical_columns or (i[2], i[3]) in numerical_columns]
    
    print('Number of rows in ground truth:', len(ground_truth))
    print('Numerical ground truth:', len(numerical_ground_truth), ':',
          len(numerical_ground_truth) * 100 / len(ground_truth), '%')
    
    print('Number of True Positives:', len(starmie_true_positives))
    print('Numerical True Positives:', len(numerical_true_postivies), ':',
          len(numerical_true_postivies) * 100 / len(starmie_true_positives), '%')

    
    print('Number of False Positives:', len(starmie_false_postives))
    print('Numerical False Positives:', len(numerical_false_postivies), ':',
          len(numerical_false_postivies) * 100 / len(starmie_false_postives), '%')


    print('Number of False Negatives:', len(starmie_false_negatives))
    print('Numerical False Negatives:', len(numerical_false_negatives), ':',
          len(numerical_false_negatives) * 100 / len(starmie_false_negatives), '%')
    
    print('Overall Precision', len(starmie_true_positives)/len(starmie_matches))
    print('Numerical Precision', len(numerical_true_postivies)/len(numerical_matches))
    print('Non Numerical Precision', len(non_numerical_true_postivies) / len(non_numerical_matches))


if __name__ == '__main__':
    main()