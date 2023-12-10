
from datetime import datetime
import os
from glob import glob
import requests

from tqdm import tqdm


def populate_dataset_graph():
    pass


def populate_pipeline_graphs(pipeline_graphs_base_dir, graphdb_endpoint, graphdb_repo):
    default_ttls = [i for i in os.listdir(pipeline_graphs_base_dir) if i.endswith('.ttl')]
    
    print(datetime.now(), ': Uploading Default and Library graphs...')
    for ttl_file in default_ttls:
        upload_file(file_path=os.path.join(pipeline_graphs_base_dir, ttl_file),
                    graphdb_endpoint=graphdb_endpoint, graphdb_repo=graphdb_repo)
    
    all_files = glob(os.path.join(pipeline_graphs_base_dir, '**', '*.ttl'), recursive=True)
    pipeline_graph_dirs = [i for i in os.listdir(pipeline_graphs_base_dir) if not i.endswith('.ttl')]
    print(datetime.now(), f': Uploading pipeline graphs: {len(all_files)} graphs for {len(pipeline_graph_dirs)} datasets ...')
    for pipeline_graph_dir in tqdm(pipeline_graph_dirs):
        pipeline_graphs = os.listdir(os.path.join(pipeline_graphs_base_dir, pipeline_graph_dir))
        for pipeline_graph in pipeline_graphs:
            upload_file(file_path=os.path.join(pipeline_graphs_base_dir, pipeline_graph_dir, pipeline_graph),
                        graphdb_endpoint=graphdb_endpoint, graphdb_repo=graphdb_repo,
                        named_graph_uri=f'http://kglids.org/resource/kaggle/{pipeline_graph_dir}/{pipeline_graph[:-4]}')
    



def upload_file(file_path, graphdb_endpoint, graphdb_repo, named_graph_uri=None):
    headers = {'Content-Type': 'application/x-turtle', 'Accept': 'application/json'}
    with open(file_path, 'rb') as f:
        file_content = f.read()
    upload_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}/statements'
    if named_graph_uri:
        upload_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}/rdf-graphs/service?graph={named_graph_uri}'

    response = requests.post(upload_url, headers=headers, data=file_content)
    if response.status_code // 100 != 2:
        print('Error uploading file:', file_path, '. Error:', response.text)
    

def main():
    graphdb_endpoint = 'http://localhost:7200'
    graphdb_repo = 'kaggle_demo'
    pipeline_graphs_base_dir = os.path.expanduser('~/projects/kglids/storage/pipeline_graphs/kaggle_demo')
    populate_pipeline_graphs(pipeline_graphs_base_dir, graphdb_endpoint, graphdb_repo)
    

if __name__ == '__main__':
    main()