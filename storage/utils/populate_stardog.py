from datetime import datetime
from glob import glob
import os
from pathlib import Path
import stardog as sd
from tqdm import tqdm

from src.util import create_pipeline_uri

def poulate_graph4code():
    connection_details = {
      'endpoint': 'http://localhost:5820',
      'username': 'admin',
      'password': 'admin'
    }
    database_name = 'g4c_1000'


    with sd.Admin(**connection_details) as admin:
        if database_name in [db.name for db in admin.databases()]:
            admin.database(database_name).drop()
        db = admin.new_database(database_name, {'edge.properties': True})

    conn = sd.Connection(database_name, **connection_details)

    conn.begin()

    nq_files = ['output_1000_graphs/'+i for i in os.listdir('output_1000_graphs') if i.endswith('nq')]
    for nq in tqdm(nq_files):
        with open(nq, 'r') as f:
            lines = f.readlines()

        if not lines:
            print('Skipping Empty Graph:', nq)
            continue
        first_line = lines[0].strip()
        graph_uri = first_line[first_line.rindex('<'):-2].strip()
        conn.add(sd.content.File(nq), graph_uri=graph_uri)
    conn.commit()



def populate_kglids():
    database_name = 'all_kaggle_datasets'
    graphs_dir = os.path.expanduser('~/projects/kglids/storage/knowledge_graph/pipeline_abstraction/all_datasets/')

    connection_details = {
      'endpoint': 'http://localhost:5820',
      'username': 'admin',
      'password': 'admin'
    }

    with sd.Admin(**connection_details) as admin:
        if database_name in [db.name for db in admin.databases()]:
            admin.database(database_name).drop()
        db = admin.new_database(database_name, {'edge.properties': True})

    conn = sd.Connection(database_name, **connection_details)

    print(datetime.now(), 'Loading Default Graph...')
    conn.begin()
    default_ttl_files = [i for i in os.listdir(graphs_dir) if i.endswith('.ttl')]
    for ttl in default_ttl_files:
        conn.add(sd.content.File(os.path.join(graphs_dir, ttl)))
    conn.commit()

    print(datetime.now(), 'Loading pipeline graphs...')
    named_graphs_dirs = [i for i in os.listdir(graphs_dir) if i not in default_ttl_files]
    for named_graph_dir in tqdm(named_graphs_dirs, smoothing=0):
        ttl_files = glob(os.path.join(graphs_dir, named_graph_dir, '**', '*.ttl'), recursive=True)
        for ttl in ttl_files:
            pipeline_url = create_pipeline_uri(source='kaggle',
                                               dataset_name=named_graph_dir,
                                               file_name=Path(ttl).stem)
            try:
                conn.begin()
                conn.add(sd.content.File(ttl), graph_uri=pipeline_url)
                conn.commit()
            except Exception as e:
                print('Error in File:', ttl)
                print(e)
                continue


if __name__ == '__main__':
    populate_kglids()
