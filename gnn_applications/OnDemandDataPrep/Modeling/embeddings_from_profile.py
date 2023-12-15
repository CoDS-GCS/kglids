import os
import csv
import json
import numpy as np
from urllib.parse import quote_plus


def get_profiles(graph_name):
    metadata = 'storage/profiles/' + graph_name
    for datatype in os.listdir(metadata):
        if datatype == '.DS_Store':
            continue
        for profile_json in os.listdir(metadata + '/' + datatype):
            if profile_json.endswith('.json'):
                with open(metadata + '/' + datatype + '/' + profile_json, 'r') as open_file:
                    yield json.load(open_file)


def generate_table_id(profile_path: str, table_name: str):
    profile_path = profile_path.split('/')
    table_name = profile_path[-1]
    dataset_name = profile_path[-3]
    table_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}'
    return table_id


def get_table_embeddings_scaling(graph_name):
    """"
    Average the embedding of the integer and float columns separately.
    Concatenate the results to obtain a 1800,1 embedding for the entire table.
    """
    table_set = set()
    string_embeddings = {}
    int_embeddings = {}
    float_embeddings = {}
    for profile in get_profiles(graph_name):
        dtype = profile['data_type']
        if dtype == 'int':
            table_key = profile['path'].split('/')[4] + '_' + profile['table_id'].split('/')[-1]
            table_set.add(table_key)

            if profile['embedding'] is not None:
                if table_key not in int_embeddings:
                    int_embeddings[table_key] = []
                int_embeddings[table_key].append(profile['embedding'])
        elif dtype == 'float':
            table_key = profile['path'].split('/')[4] + '_' + profile['table_id'].split('/')[-1]
            table_set.add(table_key)

            if profile['embedding'] is not None:
                if table_key not in float_embeddings:
                    float_embeddings[table_key] = []
                float_embeddings[table_key].append(profile['embedding'])

    profiles = {}
    for table in table_set:
        if table in int_embeddings:
            int_embeddings_avg = np.mean(int_embeddings[table], axis=0)
        else:
            int_embeddings_avg = np.zeros(300)
        if table in float_embeddings:
            float_embeddings_avg = np.mean(float_embeddings[table], axis=0)
        else:
            float_embeddings_avg = np.zeros(300)
        profiles[table] = np.concatenate(
            (int_embeddings_avg, float_embeddings_avg, np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300)))

    for key, value in profiles.items():
        if isinstance(value, np.ndarray):
            profiles[key] = value.tolist()
    file_path = "gnn_applications/OnDemandDataPrep/Modeling/storage/Embeddings_tables_"+graph_name+".csv"

    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Key'] + [f'Value_{i}' for i in range(1800)]
        writer.writerow(header)

        # Write each key-value pair as a row in the CSV file
        for key, value in profiles.items():
            writer.writerow([key] + [str(v) for v in value])


def get_table_embeddings_cleaning(graph_name):
    """"
    Average the embedding of the columns with missing values for each column type separately.
    Concatenate the results to obtain a 1800,1 embedding for the entire table.
    """
    table_set = set()
    # Separate the embeddings based on dtype
    string_embeddings = {}
    int_embeddings = {}
    float_embeddings = {}
    date_embeddings = {}
    NE_embeddings = {}
    NL_embeddings = {}

    for profile in get_profiles(graph_name):
        dtype = profile['data_type']
        missing_value = profile['missing_values_count']
        total_value = profile['total_values_count']
        if missing_value != total_value:

            table_key = profile['path'].split('/')[-2] + '--' + profile['table_id'].split('/')[
                -1]  # generate_table_id(profile['path'], profile['table_name'])
            table_set.add(table_key)

            if dtype == 'int':
                if profile['embedding'] is not None:
                    if table_key not in int_embeddings:
                        int_embeddings[table_key] = []
                    int_embeddings[table_key].append(profile['embedding'])
            elif dtype == 'float':
                if profile['embedding'] is not None:
                    if table_key not in float_embeddings:
                        float_embeddings[table_key] = []
                    float_embeddings[table_key].append(profile['embedding'])
            elif dtype == 'date':
                if profile['embedding'] is not None:
                    if table_key not in date_embeddings:
                        date_embeddings[table_key] = []
                    date_embeddings[table_key].append(profile['embedding'])
            elif dtype == 'natural_language_text':
                if profile['embedding'] is not None:
                    if table_key not in NL_embeddings:
                        NL_embeddings[table_key] = []
                    NL_embeddings[table_key].append(profile['embedding'])
            elif dtype == 'named_entity':
                if profile['embedding'] is not None:
                    if table_key not in NE_embeddings:
                        NE_embeddings[table_key] = []
                    NE_embeddings[table_key].append(profile['embedding'])
            elif dtype == 'string':
                if profile['embedding'] is not None:
                    if table_key not in string_embeddings:
                        string_embeddings[table_key] = []
                    string_embeddings[table_key].append(profile['embedding'])

    profiles = {}
    for table in table_set:
        if table in int_embeddings:  # [table]:
            int_embeddings_avg = np.mean(int_embeddings[table], axis=0)
        else:
            int_embeddings_avg = np.zeros(300)
        if table in float_embeddings:  # [table]:
            float_embeddings_avg = np.mean(float_embeddings[table], axis=0)
        else:
            float_embeddings_avg = np.zeros(300)
        if table in date_embeddings:  # [table]:
            date_embeddings_avg = np.mean(date_embeddings[table], axis=0)
        else:
            date_embeddings_avg = np.zeros(300)
        if table in NL_embeddings:  # [table]:
            NL_embeddings_avg = np.mean(NL_embeddings[table], axis=0)
        else:
            NL_embeddings_avg = np.zeros(300)
        if table in NE_embeddings:  # [table]:
            NE_embeddings_avg = np.mean(NE_embeddings[table], axis=0)
        else:
            NE_embeddings_avg = np.zeros(300)
        if table in string_embeddings:  # [table]:
            string_embeddings_avg = np.mean(string_embeddings[table], axis=0)
        else:
            string_embeddings_avg = np.zeros(300)

        profiles[table] = np.concatenate((int_embeddings_avg, float_embeddings_avg, date_embeddings_avg,
                                          NL_embeddings_avg, NE_embeddings_avg, string_embeddings_avg))

    for key, value in profiles.items():
        if isinstance(value, np.ndarray):
            profiles[key] = value.tolist()
    file_path = "gnn_applications/OnDemandDataPrep/Modeling/storage/Embeddings_tables_"+graph_name+".csv"

    #    with open(file_path, "w", newline='') as f:
    #        writer = csv.writer(f)
    #
    #        # Write each key-value pair as a row in the CSV file
    #        for key, value in profiles.items():
    #            writer.writerow([key, value])
    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Key'] + [f'Value_{i}' for i in range(1800)]
        writer.writerow(header)

        # Write each key-value pair as a row in the CSV file
        for key, value in profiles.items():
            writer.writerow([key] + [str(v) for v in value])


def get_column_embeddings(graph_name):
    embeddings = {}
    for profile in get_profiles(graph_name):
        dtype = profile['data_type']

        column_key = profile['path'].split('/')[4] + '--' + profile['table_id'].split('/')[-1] + '--' + \
                     profile['column_id'].split('/')[-1]

        if profile['embedding'] is not None:
            if dtype == 'int':
                embeddings[column_key] = np.concatenate(
                    (profile['embedding'], np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300)))
            elif dtype == 'float':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), profile['embedding'], np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300)))
            elif dtype == 'date':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), np.zeros(300), profile['embedding'], np.zeros(300), np.zeros(300), np.zeros(300)))
            elif dtype == 'natural_language_text':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), np.zeros(300), np.zeros(300), profile['embedding'], np.zeros(300), np.zeros(300)))
            elif dtype == 'named_entity':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), profile['embedding'], np.zeros(300)))
            elif dtype == 'string':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), profile['embedding']))
            elif dtype == 'boolean':
                embeddings[column_key] = np.concatenate(
                    (np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300), np.zeros(300)))

    file_path = 'gnn_applications/OnDemandDataPrep/Modeling/storage/Embeddings_columns_' + graph_name + '.csv'

    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Key'] + [f'Value_{i}' for i in range(1800)]
        writer.writerow(header)

        # Write each key-value pair as a row in the CSV file
        for key, value in embeddings.items():
            writer.writerow([key] + [str(v) for v in value])


def get_column_embeddings_unary(graph_name):
    embeddings = {}
    for profile in get_profiles(graph_name):
        column_key = profile['path'].split('/')[5] + '_' + profile['table_id'].split('/')[-1][4:] + '_' + \
                     profile['column_id'].split('/')[-1]

        if profile['embedding'] is not None:
            embeddings[column_key] = profile['embedding']

    file_path = "gnn_applications/OnDemandDataPrep/Modeling/storage/Embeddings_columns_"+graph_name+".csv"

    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = ['Key'] + [f'Value_{i}' for i in range(300)]
        writer.writerow(header)

        for key, value in embeddings.items():
            writer.writerow([key] + [str(v) for v in value])
