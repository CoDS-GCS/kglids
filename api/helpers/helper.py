import io
import stardog
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, CSV


def connect_to_stardog(endpoint: str, port, db_name: str = 'kglids'):
    connection_details = {
        'endpoint': 'http://{}:{}'.format(endpoint, str(port)),
        'username': 'admin',
        'password': 'admin'
    }
    print('Connected to Stardog: https://cloud.stardog.com/')
    return stardog.Connection(db_name, **connection_details)


def execute_query(conn: stardog.Connection, query: str, return_type: str = 'csv'):
    if return_type == 'csv':
        result = conn.select(query, content_type='text/csv')
        return pd.read_csv(io.BytesIO(result))
    elif return_type == 'json':
        result = conn.select(query)
        return result['results']['bindings']
    else:
        raise ValueError(return_type, ' not supported')


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