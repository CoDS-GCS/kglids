from datetime import datetime
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

import nbconvert
import pandas as pd
import psycopg
from tqdm import tqdm


def main():
    target_competitions = ['santander-customer-transaction-prediction', 'home-credit-default-risk']
    target_datasets = []  # TODO: add support for datasets
    meta_kaggle_code_path = os.path.expanduser('~/data/meta_kaggle_code')
    meta_kaggle_code_files = glob.glob(os.path.join(meta_kaggle_code_path, '**', '*.*'), recursive=True)
    output_files_path = os.path.expanduser('~/projects/kglids/storage/pipelines/top_2_competitions')

    notebook_id_to_path = {int(Path(file).stem): file for file in meta_kaggle_code_files}

    meta_kaggle_db_creds = {'dbname': 'meta_kaggle', 'host': 'localhost', 'port': 5432,
                            'user': 'admin', 'password': 'admin'}

    query_competition_filter = f"""AND c."Slug" in ('{"', '".join(target_competitions)}')"""
    query = """
            SELECT 
                kv."Id" AS kernel_version_id, 
                kv."ScriptId" AS kernel_id,
                kv."Title" AS title,
                kv."CreationDate" AS creation_date,
                MAX(c."Slug") AS competition,
                MAX(u."UserName") AS author,
                MAX(k."CurrentUrlSlug") AS kernel_name,
                MAX(k."TotalVotes") AS votes,
                MAX(s."PublicScoreLeaderboardDisplay") AS score
            FROM 
                kernelversion kv 
                JOIN kernel k ON kv."ScriptId" = k."Id" 
                JOIN kernellanguage kl ON kv."ScriptLanguageId" = kl."Id"
                JOIN kernelversioncompetitionsource kvcs ON kv."Id" = kvcs."KernelVersionId"
                JOIN competitions c ON  kvcs."SourceCompetitionId" = c."Id"
                JOIN users u ON kv."AuthorUserId" = u."Id"
                LEFT JOIN submission s ON kv."Id" = s."SourceKernelVersionId"	-- kernel versions might have multiple scores
            WHERE 
                kv."Id" = k."CurrentKernelVersionId" -- latest version of each notebook only
                AND kl."Id" in (2, 9) -- Python files and notebooks
                {}
            GROUP BY
                kv."Id"
    """.format(query_competition_filter)

    # 1 run query against DB
    print(datetime.now(), ': Querying MetaKaggle Database for', len(target_competitions), 'competitions.')
    with psycopg.connect(**meta_kaggle_db_creds) as conn:
        kernels_df = pd.read_sql_query(query, conn)

    # 2 copy the meta kaggle code notebooks to output dir and convert file to .py (if ipynb)
    print(datetime.now(), ': Loading and converting', len(kernels_df), 'notebooks.')
    os.makedirs(output_files_path, exist_ok=True)

    pool = mp.Pool()
    rows = kernels_df.apply(lambda row: (row['kernel_version_id'], row['author'], row['kernel_name'],
                                         row['title'], row['votes'], row['score'], row['creation_date'],
                                         row['competition'], notebook_id_to_path, output_files_path),
                            axis=1).tolist()
    # list(tqdm(pool.imap_unordered(convert_and_save_notebook, rows), total=len(rows)))
    # TODO: multiprocessing takes longer time. Why?
    for row in tqdm(rows):
        convert_and_save_notebook(row)
    print(datetime.now(), ': Done.')


def convert_and_save_notebook(args):
    (kernel_version_id, author, kernel_name, title, votes, score, creation_date, competition,
     notebook_id_to_path, output_files_path) = args
    if kernel_version_id not in notebook_id_to_path:
        return

    notebook_path = notebook_id_to_path[kernel_version_id]
    with open(notebook_path, 'r') as f:
        if notebook_path.endswith('.py'):
            source_code = f.read()
        else:
            notebook_exporter = nbconvert.PythonExporter()
            source_code, _ = notebook_exporter.from_file(f)

    metadata = {'url': f"https://www.kaggle.com/code/{author}/{kernel_name}",
                'title': title,
                'author': author,
                'votes': votes,
                'score': score if score else None,  # random.uniform(0.1, 1.0),
                'date': creation_date
                }

    # save the python file and metadata in corresponding directory under dataset
    notebook_save_path = os.path.join(output_files_path, competition,
                                      f'{author}.{kernel_name}')
    os.makedirs(notebook_save_path, exist_ok=True)
    with open(os.path.join(notebook_save_path, kernel_name + '.py'), 'w') as f:
        f.write(source_code)
    with open(os.path.join(notebook_save_path, 'pipeline_info.json'), 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    main()
