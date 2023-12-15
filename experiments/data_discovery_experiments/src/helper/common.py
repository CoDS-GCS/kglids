import pandas as pd


def get_table_mapping(df: pd.DataFrame):
    print("getting mapping between tables from ground truth")
    query_tables = df[df.columns[0]].to_list()
    candidate_tables = df[df.columns[1]].to_list()
    mapping = {(query_tables[i], candidate_tables[i]) for i in range(len(query_tables))}

    return mapping


def load_groundtruth(dataset):
    df = 'null'
    if dataset == 'smallerReal':
        file = 'ds_gt.csv'
        print('loading {}'.format(file))
        df = pd.read_csv('../gt_files/' + file)
        df[df.columns[0]] = df[df.columns[0]] + '.csv'
        df[df.columns[1]] = df[df.columns[1]] + '.csv'
    elif dataset == 'synthetic':
        file = 'alignment_groundtruth.csv'
        print('loading {}'.format(file))
        df = pd.read_csv('../gt_files/' + file)
    elif dataset == 'TUS':
        df = pd.read_csv('../gt_files/tus_groundtruth.csv')
    
    return df