from SPARQLWrapper import SPARQLWrapper


def connect_to_blazegraph(port, namespace):
    endpoint = 'http://localhost:{}/blazegraph/namespace/'.format(port) + namespace + '/sparql'
    print('Connected to {}'.format(endpoint))
    return SPARQLWrapper(endpoint)
