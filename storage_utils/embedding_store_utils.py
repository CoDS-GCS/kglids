from datetime import datetime
import fasttext
from glob import glob
import json
import math
import os
from tqdm import tqdm
import psycopg

from kglids_config import KGLiDSConfig


def create_embedding_db(db_name):
    try:
        conn = psycopg.connect(dbname='postgres', user='postgres', password='postgres', autocommit=True)

        cursor = conn.cursor()
        cursor.execute(f'DROP DATABASE IF EXISTS {db_name};')
        cursor.execute(f'CREATE DATABASE {db_name};')
        conn.close()

        conn = psycopg.connect(dbname=db_name, user='postgres', password='postgres', autocommit=True)
        cursor = conn.cursor()
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        cursor.execute(f'''CREATE TABLE {db_name} (
        id text primary key,
        name text,
        data_type text,
        dataset_name text,
        table_name text,
        content_embedding vector(300),
        label_embedding vector(300),
        content_label_embedding vector(600)
        );''')

        cursor.execute(f'CREATE INDEX ON {db_name} (data_type);')
        cursor.execute(f'CREATE INDEX ON {db_name} (dataset_name);')
        cursor.execute(f'CREATE INDEX ON {db_name} (table_name);')
        cursor.execute(f'CREATE INDEX ON {db_name} USING hnsw (content_embedding vector_cosine_ops);')
        cursor.execute(f'CREATE INDEX ON {db_name} USING hnsw (label_embedding vector_cosine_ops);')
        cursor.execute(f'CREATE INDEX ON {db_name} USING hnsw (content_label_embedding vector_cosine_ops);')

        conn.close()
        print(datetime.now(), ': Database Created.')
    except psycopg.Error as e:
        print(datetime.now(), ': Error creating database:', e)


def insert_columns(column_data, db_name):
    insert_query = f'''INSERT INTO {db_name} (id, name, data_type, dataset_name, table_name, 
                        content_embedding, label_embedding, content_label_embedding) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING;'''

    conn = psycopg.connect(dbname=db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    cursor.executemany(insert_query, column_data)
    cursor.close()


def populate_column_embeddings(column_profiles_path, embedding_db_name):

    fasttext_path = os.path.join(KGLiDSConfig.base_dir, 'storage/embeddings/cc.en.300.bin')
    batch_size = 1000

    print(datetime.now() ,': Loading Fasttext embeddings...')
    ft = fasttext.load_model(fasttext_path)

    # go over column profiles:
    print(datetime.now(), ': Populating vector database with column info')
    profile_paths = glob(os.path.join(column_profiles_path, '**', '*.json'), recursive=True)
    batch = []
    for profile_path in tqdm(profile_paths):
        with open(profile_path) as f:
            profile = json.load(f)

        if profile['embedding']:
            content_embedding = profile['embedding'] #+ [0] + [math.sqrt(profile['embedding_scaling_factor'])]
        else:
            # boolean column
            content_embedding = [profile['true_ratio']] * 300

        # generate name embeddings and name + value embeddings
        sanitized_name = profile['column_name'].replace('\n', ' ').replace('_', ' ').strip()
        label_embedding = ft.get_sentence_vector(sanitized_name).tolist()
        content_label_embedding = content_embedding + label_embedding

        # store for each column: name, type, id, dataset, table, value embed, name embed, name+value embed
        batch.append((profile['column_id'], profile['column_name'], profile['data_type'],
                      profile['dataset_name'], profile['table_name'],
                      str(content_embedding), str(label_embedding), str(content_label_embedding)))

        if len(batch) >= batch_size:
            insert_columns(batch, embedding_db_name)
            batch = []
    if batch:
        insert_columns(batch, embedding_db_name)

    print(datetime.now(), ': Database populated with', len(profile_paths), 'columns.')
