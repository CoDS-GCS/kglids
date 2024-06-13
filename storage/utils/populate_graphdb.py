
from datetime import datetime
import os
from glob import glob
import requests
from urllib.parse import quote
import json

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
            pipeline_graph_path = os.path.join(pipeline_graphs_base_dir, pipeline_graph_dir, pipeline_graph)
            with open(pipeline_graph_path, 'r') as f:
                for line in f:
                    if 'a kglids:Statement' in line:
                        # sanitize the URL so it has the correct IRI on GraphDB (needs to be url encoded twice)
                        named_graph_uri_raw = '/'.join(line.split()[0].replace('<http://kglids.org/', '').split('/')[:-1])
                        break
            named_graph_uri = 'http://kglids.org/' + quote(named_graph_uri_raw)

            upload_file(file_path=pipeline_graph_path,
                        graphdb_endpoint=graphdb_endpoint, graphdb_repo=graphdb_repo,
                        named_graph_uri=named_graph_uri)
    

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
    

def create_or_replace_repo(graphdb_endpoint, graphdb_repo):
    # check if repo with the same name exists
    url = graphdb_endpoint + '/rest/repositories'
    graphdb_repos = json.loads(requests.get(url).text)
    graphdb_repo_ids = [i['id'] for i in graphdb_repos]

    # remove existing repo if found
    if graphdb_repo in graphdb_repo_ids:
        url = f"{graphdb_endpoint}/rest/repositories/{graphdb_repo}"
        response = requests.delete(url)
        if response.status_code // 100 != 2:
            print(datetime.now(), ': Error while deleting GraphDB repo:', graphdb_repo, ':', response.text)

    # create a new repo
    url = graphdb_endpoint + '/rest/repositories'
    headers = {"Content-Type": "application/json"}
    data = {
        "id": graphdb_repo,
        "type": "graphdb",
        "title": graphdb_repo,
        "params": {
            "defaultNS": {
                "name": "defaultNS",
                "label": "Default namespaces for imports(';' delimited)",
                "value": ""
            },
            "imports": {
                "name": "imports",
                "label": "Imported RDF files(';' delimited)",
                "value": ""
            },
            "enableContextIndex": {
                "name": "enableContextIndex",
                "label": "Enable context index",
                "value": "true"
            }
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code // 100 != 2:
        print(datetime.now(), "Error creating the GraphDB repo:", graphdb_repo, ':', response.text)

def main():
    graphdb_endpoint = 'http://localhost:7200'
    graphdb_repo = 'kaggle_demo'
    pipeline_graphs_base_dir = os.path.expanduser('~/projects/kglids/storage/pipeline_graphs/kaggle_demo')
    populate_pipeline_graphs(pipeline_graphs_base_dir, graphdb_endpoint, graphdb_repo)
    

if __name__ == '__main__':
    main()