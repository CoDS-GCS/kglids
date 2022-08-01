import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import pickle


import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
from SPARQLWrapper import SPARQLWrapper, JSON
import ampligraph
from ampligraph.latent_features import ComplEx, TransE, DistMult, HolE


def fetch_graphs(graph_uris, graphs_dir, sparql_endpoint):

    for uri in tqdm(graph_uris):
        query = """ select  ?subject ?predicate ?object
                     where {
                       graph <PLACEHOLDER> {?subject ?predicate ?object}
                     }
                  """.replace('PLACEHOLDER', uri)

        sparql = SPARQLWrapper(sparql_endpoint, returnFormat=JSON)
        sparql.setCredentials('admin', 'admin')
        sparql.setQuery(query)
        results = sparql.query().convert()
        spo = []
        for result in results["results"]["bindings"]:
            if result['subject']["type"] == 'uri':
                subj, pred, obj = result['subject']['value'], result['predicate']["value"].strip(), result['object'][
                    "value"]
            else:
                subj, pred, obj = result['subject']['s']['value'], result['subject']['p']['value'], \
                                  result['subject']['o']['value']
            spo.append([subj, pred, obj])
        df = pd.DataFrame(spo, columns=['s', 'p', 'o'])
        file_path = os.path.join(graphs_dir, uri.lstrip('http://kglids.org/resource/kaggle/') + '.tsv')
        os.makedirs(Path(file_path).parent, exist_ok=True)
        df.to_csv(file_path, index=False, sep='\t')


def generate_embeddings(graph_uris, graphs_dir, embedding_size=100):
    for uri in tqdm(graph_uris):
        file_path = os.path.join(graphs_dir, uri.lstrip('http://kglids.org/resource/kaggle/') + '.tsv')
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, delimiter='\t').astype(str)
        if not len(df):
            print('Removing Empty Pipeline:', uri)
            os.remove(file_path)
            continue
        nodes = list(set(df['s'].tolist() + df['o'].tolist()))
        transe_model = TransE(batches_count=10, epochs=20, k=embedding_size, eta=20, optimizer='adam',
                              optimizer_params={'lr': 1e-2}, loss='multiclass_nll', regularizer='LP',
                              regularizer_params={'p': 3, 'lambda': 1e-2})
        complex_model = ComplEx(batches_count=10, epochs=20, k=int(embedding_size/2), eta=20,
                                optimizer='adam', optimizer_params={'lr': 1e-2}, loss='multiclass_nll',
                                regularizer='LP', regularizer_params={'p': 3, 'lambda': 1e-2})
        try:
            transe_model.fit(df.to_numpy())
            complex_model.fit(df.to_numpy())
        except:
            print('Removing Troublesome pipeline:', uri)
            os.remove(file_path)
            continue
        transe_embeddings = transe_model.get_embeddings(nodes)
        complex_embeddings = complex_model.get_embeddings(nodes)
        embeddings = {}
        for i in range(len(nodes)):
            embeddings[nodes[i]] = {'transE': transe_embeddings[i], 'complEx': complex_embeddings[i]}

        with open(file_path.replace('.tsv', '.pickle'), 'wb') as f:
            pickle.dump(embeddings, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_uris', default='task1_uris_labels.csv') # task1_uris_labels.csv or task2_uris_labels.csv ...
    parser.add_argument('--graph_out_dir', default='task1_kglids_graphs') # task1_kglids_graphs or task1_graph4code_graphs ...
    parser.add_argument('--endpoint', default='kglids') #kglids or graph4code
    args = parser.parse_args()

    graph_uris_labels = pd.read_csv(args.graph_uris)
    endpoint = f'http://localhost:5858/{args.endpoint}/query'

    # graph_uris_labels = pd.read_csv('task1_uris_labels.csv')
    # kglids_endpoint = 'http://localhost:5858/kglids/query'
    # graph4code_endpoint = 'http://localhost:5858/graph4code/query'

    ## fetch_graphs(graph_uris_labels.uri.tolist(), 'kglids_graphs', kglids_endpoint)
    ## generate_embeddings(graph_uris_labels.uri.tolist(), 'kglids_graphs')

    fetch_graphs(graph_uris_labels.uri.tolist(), args.graph_out_dir, endpoint)
    generate_embeddings(graph_uris_labels.uri.tolist(), args.graph_out_dir)


if __name__ == '__main__':
    main()
