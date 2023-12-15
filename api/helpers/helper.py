import io
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, CSV


def connect_to_graphdb(endpoint, graphdb_repo):

    graphdb = SPARQLWrapper(f'{endpoint}/repositories/{graphdb_repo}')    
    return graphdb


def query_graphdb(graphdb_conn: SPARQLWrapper, query, return_type='csv'):
    
    graphdb_conn.setQuery(query)
    if return_type == 'csv':
        graphdb_conn.setReturnFormat(CSV)
        results = graphdb_conn.queryAndConvert()
        return pd.read_csv(io.BytesIO(results))
    elif return_type == 'json':
        graphdb_conn.setReturnFormat(JSON)
        results = graphdb_conn.queryAndConvert()
        return results['results']['bindings']
    else:
        raise ValueError(return_type, ' not supported')