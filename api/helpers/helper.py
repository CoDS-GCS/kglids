import SPARQLWrapper.Wrapper
import io
import stardog
import pandas as pd
from SPARQLWrapper import JSON


def connect_to_stardog(port, db_name: str = 'kglids'):
    connection_details = {
        'endpoint': 'http://localhost:{}'.format(str(port)),
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


def execute_query_blazegraph(sparql: SPARQLWrapper.Wrapper.SPARQLWrapper, query: str):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
