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
    INTPUT_GROUND_TRUTH_PATH = '/home/mossad/projects/santos/groundtruth/santosUnionBenchmark.pickle'
    OUTPUT_FILE_PATH = '/home/mossad/projects/kglids/storage/data_sources/dataset_storage/experiments/santos/groundtruth/groundtruth.csv'

    ground_truth = pd.read_pickle(INTPUT_GROUND_TRUTH_PATH)
    
    ground_truth_table = []
    for table, related_tables in ground_truth.items():
        for related_table in related_tables:
            ground_truth_table.append([table, related_table, 1])
    columns = ['target_ds', 'candidate_ds', 'attr_pairs']
    ground_truth_df = pd.DataFrame(ground_truth_table, columns=columns)
    ground_truth_df.to_csv(OUTPUT_FILE_PATH, index=False)

if __name__ == '__main__':
    main()