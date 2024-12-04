from datetime import datetime
from glob import glob
import json
import requests
import os
import shutil
import time
from tqdm import tqdm
from urllib.parse import quote

from kglids_config import KGLiDSConfig


def create_graphdb_repo(graphdb_repo_name, graphdb_endpoint):
    # check if repo with the same name exists
    url = graphdb_endpoint + '/rest/repositories'
    graphdb_repos = json.loads(requests.get(url).text)
    graphdb_repo_ids = [i['id'] for i in graphdb_repos]
    # remove existing repo if found
    headers = {"Content-Type": "application/json"}
    if graphdb_repo_name in graphdb_repo_ids:
        if KGLiDSConfig.replace_existing_graphdb_repo:
            url = f"{graphdb_endpoint}/rest/repositories/{graphdb_repo_name}"
            response = requests.delete(url)
            if response.status_code // 100 != 2:
                print(datetime.now(), ': Error while deleting GraphDB repo:', graphdb_repo_name, ':',
                      response.text)
    else:
        # create a new repo
        url = graphdb_endpoint + '/rest/repositories'
        data = {
            "id": graphdb_repo_name,
            "type": "graphdb",
            "title": graphdb_repo_name,
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
            print(datetime.now(), "Error creating the GraphDB repo:", graphdb_repo_name, ':',
                  response.text)
        else:
            print(datetime.now(), "Created GraphDB repo:", graphdb_repo_name)


def populate_pipeline_graphs(pipeline_graphs_base_dir, graphdb_endpoint, graphdb_repo):
    default_ttls = [i for i in os.listdir(pipeline_graphs_base_dir) if i.endswith('.ttl')]

    print(datetime.now(), ': Uploading Default and Library graphs...')
    for ttl_file in default_ttls:
        _upload_graph(file_path=os.path.join(pipeline_graphs_base_dir, ttl_file),
                      graphdb_endpoint=graphdb_endpoint, graphdb_repo=graphdb_repo)

    all_files = glob(os.path.join(pipeline_graphs_base_dir, '**', '*.ttl'), recursive=True)
    pipeline_graph_dirs = [i for i in os.listdir(pipeline_graphs_base_dir) if not i.endswith('.ttl')]
    print(datetime.now(),
          f': Uploading pipeline graphs: {len(all_files)} graphs for {len(pipeline_graph_dirs)} datasets ...')
    for pipeline_graph_dir in tqdm(pipeline_graph_dirs):
        pipeline_graphs = os.listdir(os.path.join(pipeline_graphs_base_dir, pipeline_graph_dir))
        for pipeline_graph in pipeline_graphs:
            pipeline_graph_path = os.path.join(pipeline_graphs_base_dir, pipeline_graph_dir, pipeline_graph)
            with open(pipeline_graph_path, 'r') as f:
                for line in f:
                    if 'a kglids:Statement' in line:
                        # sanitize the URL so it has the correct IRI on GraphDB (needs to be url encoded twice)
                        named_graph_uri_raw = '/'.join(
                            line.split()[0].replace('<http://kglids.org/', '').split('/')[:-1])
                        break
            named_graph_uri = 'http://kglids.org/' + quote(named_graph_uri_raw)

            _upload_graph(file_path=pipeline_graph_path,
                          graphdb_endpoint=graphdb_endpoint, graphdb_repo=graphdb_repo,
                          named_graph_uri=named_graph_uri)


def _upload_graph(file_path, graphdb_endpoint, graphdb_repo, named_graph_uri=None):
    headers = {'Content-Type': 'application/x-turtle', 'Accept': 'application/json'}
    with open(file_path, 'rb') as f:
        file_content = f.read()
    upload_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}/statements'
    if named_graph_uri:
        upload_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}/rdf-graphs/service?graph={named_graph_uri}'

    response = requests.post(upload_url, headers=headers, data=file_content)
    if response.status_code // 100 != 2:
        print('Error uploading file:', file_path, '. Error:', response.text)


def populate_data_global_schema_graph(data_global_schema_graph_path, graphdb_repo_name, graphdb_endpoint,
                                      graphdb_import_path):
    # copy generated file to graphdb-import
    tmp_file_name = graphdb_repo_name + '_import.ttl'
    shutil.copy2(data_global_schema_graph_path, os.path.join(graphdb_import_path, tmp_file_name))
    # import copied graph
    url = f"{graphdb_endpoint}/rest/repositories/{graphdb_repo_name}/import/server"
    headers = {"Content-Type": "application/json"}
    data = {"fileNames": [tmp_file_name]}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code // 100 != 2:
        print(datetime.now(), 'Error importing file:', tmp_file_name, 'to GraphDB repo:', graphdb_repo_name,
              ':',
              response.text)

    # check whether the graph is imported successfully
    import_completed = False
    while not import_completed:
        time.sleep(5)  # Wait for 5 seconds before checking again
        status_url = f"{graphdb_endpoint}/rest/repositories/{graphdb_repo_name}/import/server/{response.json()['taskId']}"
        status_response = requests.get(status_url, headers=headers)
        if status_response.status_code == 200:
            status = status_response.json()['status']
            if status == 'FINISHED':
                import_completed = True
                print(datetime.now(), "Graph import completed successfully!")
            elif status == 'FAILED':
                print(datetime.now(), "Graph import failed!")
                break
            else:
                print(datetime.now(), "Graph import in progress...")
        else:
            print(datetime.now(), "Error checking import status:", status_response.text)

    # remove copied graph
    if import_completed:
        os.remove(os.path.join(graphdb_import_path, tmp_file_name))
