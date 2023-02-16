import pickle
from collections import defaultdict

import pandas as pd


def main():
    """
    Takes a ground truth table (csv file in D3L format), and reformats it for Santos (pickle file in Santos format).
    Input Format: CSV file: table name, table name, number of related attribute pairs
                                target_ds candidate_ds  attr_pairs
                                pHNPKRH      iHUdrTj          10
    Output Format: Dict[Str, List]. Dictionary from table name to a list of names of related tables.
    """
    INTPUT_GROUND_TRUTH_PATH = '/home/mossad/projects/kglids/storage/data_sources/dataset_storage/experiments/smallerReal/groundtruth/ds_gt.csv'
    OUTPUT_FILE_PATH = '/home/mossad/projects/santos/groundtruth/samllerRealUnionBenchmark.pickle'
    
    ground_truth = pd.read_csv(INTPUT_GROUND_TRUTH_PATH)
    
    related_tables_dict = defaultdict(set)
    for table1, table2 in zip(ground_truth['target_ds'].values, ground_truth['candidate_ds'].values):
        related_tables_dict[table1 + '.csv'].add(table2 + '.csv')
        related_tables_dict[table2 + '.csv'].add(table1 + '.csv')

    output_dict = {}
    for k in related_tables_dict.keys():
        output_dict[k] = list(related_tables_dict[k])
    
    with open(OUTPUT_FILE_PATH, 'wb') as f:
        pickle.dump(output_dict, f)



if __name__ == '__main__':
    main()