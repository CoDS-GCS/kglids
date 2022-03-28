import pandas as pd
from data_items.kglids_evaluations.src.helper.queries import execute_query
from data_items.kglids_evaluations.src.helper.queries import get_top_k_tables

PREFIXES = """
    PREFIX kglids: <http://kglids.org/ontology/>
    PREFIX data:   <http://kglids.org/ontology/data/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    """


def get_datasets(config, show_query):
    query = PREFIXES + """
    SELECT ?dataset_name
    WHERE
    {
        ?dataset_id	rdf:type	kglids:Dataset	.
        ?dataset_id schema:name	?dataset_name	.
    }"""
    if show_query:
        print(query)

    result = []
    res = execute_query(config, query)
    for r in res["results"]["bindings"]:
        result.append(r["dataset_name"]["value"])
    return pd.DataFrame({"Dataset": result})


def get_tables(config, dataset: str, show_query):
    if dataset:
        dataset = '?dataset_id schema:name "{}" .'.format(dataset)
    query = PREFIXES + """
    SELECT ?table_name ?dataset_name ?table_path
    WHERE
    {
        ?table_id	rdf:type            kglids:Table	.
        %s
        ?table_id	kglids:isPartOf     ?dataset_id		.
        ?table_id  	schema:name			?table_name		.
        ?table_id	data:hasFilePath	?table_path		.
        ?dataset_id	schema:name			?dataset_name	.	
    }""" % dataset
    if show_query:
        print(query)

    tables = []
    datasets = []
    paths = []
    res = execute_query(config, query)
    for r in res["results"]["bindings"]:
        tables.append(r["table_name"]["value"])
        datasets.append(r["dataset_name"]["value"])
        paths.append(r["table_path"]["value"])

    return pd.DataFrame({'Table': tables, 'Dataset': datasets, 'Path_to_table': paths})


def recommend_tables(config, table: str, k: int, show_query: bool):
    query = PREFIXES + """
    SELECT ?table_name1 ?table_name2 ?certainty
    WHERE
    {
        ?table_id	schema:name		"%s"	        .
      	?table_id	schema:name		?table_name1	.
      	?column_id	kglids:isPartOf ?table_id		.
      	
      	<<?column_id data:hasSemanticSimilarity	?column_id2>>	data:withCertainty	?certainty	. 
      	
      	FILTER (?certainty >= 0.75)					.
      	?column_id2 kglids:isPartOf	?table_id2		.
      	?table_id2	schema:name		?table_name2	.
    }
    """ % table
    if show_query:
        print(query)

    res = execute_query(config, query)
    result = []
    for r in res["results"]["bindings"]:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        certainty = float(r["certainty"]["value"])
        result.append([(table1, table2), certainty])

    result = get_top_k_tables(result)[:k]
    result = list(map(lambda x: x[1], result))
    return pd.DataFrame({'Recommended_table': result})


def show_graph_info(config, show_query: bool):
    def count_nodes(node: str):
        return PREFIXES + """
        SELECT (COUNT(?n_%s) as ?total_number_of_%s)
        WHERE
        {
            ?n_%s rdf:type 	kglids:%s	.
        }
        """ % (node, node, node, node.capitalize())

    result = []
    for i in ['dataset', 'table', 'column']:
        if show_query:
            count_nodes(i)
        res = execute_query(config, count_nodes(i))
        for r in res["results"]["bindings"]:
            result.append(r["total_number_of_{}".format(i)]["value"])

    return pd.DataFrame({'Total_datasets': [result[0]], 'Total_tables': [result[1]],
                         'Total_columns': [result[2]], 'Total_pipelines': ['Not yet supported!']})


def _create_tables_df_row(results):
    return {
        'table_name': results['name']['value'],
        'dataset_name': results['dataset_name']['value'],
        'number_of_columns': float(results['number_of_columns']['value']),
        'number_of_rows': float(results['number_of_rows']['value']),
        'path': results['path']['value']
    }


def search_tables_on(config, all_conditions: tuple, show_query: bool):
    def search(conditions: tuple):
        return PREFIXES + \
               '\nselect ?name ?dataset_name ?path (' \
               'count(distinct ?cols) as ?number_of_columns) (max (?total) as ?number_of_rows)' \
               '\nwhere {' \
               '\n?table schema:name ?name.' \
               '\n?table data:hasFilePath ?path.' \
               '\n?table kglids:isPartOf ?dataset.' \
               '\n?dataset schema:name ?dataset_name.' \
               '\n?cols kglids:isPartOf ?table.' \
               '\n?cols data:hasTotalValueCount ?total.\n' \
               + conditions[0] + \
               '\nfilter( ' + conditions[1] + ')}' \
                                              '\n group by ?name ?dataset_name ?path'
    query = search(all_conditions)
    if show_query:
        print(query)
    res = execute_query(config, query)
    bindings = res["results"]["bindings"]

    for result in bindings:
        yield _create_tables_df_row(result)
