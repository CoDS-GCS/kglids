from SPARQLWrapper import SPARQLWrapper, JSON

def connect_to_blazegraph(namespace):
    endpoint = 'http://localhost:9999/blazegraph/namespace/' + namespace + '/sparql'
    print('connected to blazegraph ({})'.format(endpoint))
    return SPARQLWrapper(endpoint)