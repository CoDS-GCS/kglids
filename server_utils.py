from SPARQLWrapper import SPARQLWrapper, JSON

def query_graph(graph_query, graphdb_endpoint):

  try:
    sparql = SPARQLWrapper(graphdb_endpoint)
    sparql.setQuery(graph_query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30)
    results = sparql.queryAndConvert()
    return results['results']['bindings']
  except:
    return []