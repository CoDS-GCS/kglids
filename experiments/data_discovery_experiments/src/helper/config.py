import stardog
from SPARQLWrapper import SPARQLWrapper


def connect_to_blazegraph(port, namespace):
    endpoint = 'http://localhost:{}/blazegraph/namespace/'.format(port) + namespace + '/sparql'
    print('Connected to {}'.format(endpoint))
    return SPARQLWrapper(endpoint)


def connect_to_stardog(db: str, port: int = 5820):
    connection_details = {
        'endpoint': 'http://localhost:{}'.format(str(port)),
        'username': 'admin',
        'password': 'admin'
    }
    return stardog.Connection(db, **connection_details)
