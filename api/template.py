import pandas as pd
import zlib
import re
import numpy as np
import seaborn as sns
from graphviz import Digraph
from camelsplit import camelsplit
from matplotlib import pyplot as plt
from kg_governor.data_global_schema_builder.src.utils.utils import Label
from api.helpers.helper import query_graphdb

PREFIXES = """
    PREFIX kglids: <http://kglids.org/ontology/>
    PREFIX data:   <http://kglids.org/ontology/data/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
    PREFIX lib: <http://kglids.org/resource/library/> 
    """


def query_kglids(config, rdf_query):
    return query_graphdb(config, PREFIXES + rdf_query)


def get_datasets_info(config, show_query):
    query = PREFIXES + """
    SELECT ?Dataset (count(?table_id) as ?Number_of_tables)
    WHERE
    {
        ?dataset_id	rdf:type	kglids:Dataset	.
        ?dataset_id schema:name	?Dataset	    .
      	?table_id	kglids:isPartOf	?dataset_id	.
    }
    group by ?Dataset """
    if show_query:
        print(query)
    return query_graphdb(config, query)


def get_tables_info(config, dataset: str, show_query):
    if dataset:
        dataset = '?dataset_id schema:name "{}" .'.format(dataset)
    query = PREFIXES + """
    SELECT ?Table ?Dataset ?Path_to_table
    WHERE
    {
        ?table_id	rdf:type            kglids:Table	.
        %s
        ?table_id	kglids:isPartOf     ?dataset_id		.
        ?table_id  	schema:name			?Table  		.
        ?table_id	data:hasFilePath	?Path_to_table	.
        ?dataset_id	schema:name			?Dataset	    .	
    }""" % dataset
    if show_query:
        print(query)
    return query_graphdb(config, query)


def get_top_k_tables(pairs: list):
    top_k = {}
    dataset = {}
    path = {}
    for p in pairs:
        if p[0] not in top_k:
            top_k[p[0]] = p[1]
            dataset[p[0]] = p[2]
            path[p[0]] = p[3]
        else:
            updated_score = top_k.get(p[0]) + p[1]
            top_k[p[0]] = updated_score

    scores = top_k
    top_k = list(dict(sorted(top_k.items(), key=lambda item: item[1], reverse=True)).keys())
    top_k = [list(ele) for ele in top_k]

    for pair in top_k:
        c1 = pair[0]
        c2 = pair[1]
        pair = pair.extend([scores.get((c1, c2)), dataset.get((c1, c2)), path.get((c1, c2))])

    return top_k


def recommend_tables(config, dataset: str, table: str, k: int, relation: str, show_query: bool):
    query = PREFIXES + """
    SELECT ?table_name1 ?table_name2 ?certainty ?dataset2_n ?path
    WHERE
    {
        ?table_id	schema:name		"%s"	        .
      	?table_id	schema:name		?table_name1	.
      	?dataset_id schema:name     "%s"            .
      	?table_id   kglids:isPartOf ?dataset_id     .
      	?column_id	kglids:isPartOf ?table_id		.

      	<<?column_id %s	?column_id2>>	data:withCertainty	?certainty	. 
      	?column_id2 kglids:isPartOf	?table_id2		.
      	?table_id2	schema:name		?table_name2	.
      	?table_id2  data:hasFilePath ?path          .
      	?table_id2  kglids:isPartOf ?dataset2       .
      	?dataset2   schema:name     ?dataset2_n     .
    }
    """ % (table, dataset, relation)
    if show_query:
        print(query)

    res = query_graphdb(config, query, return_type='json')
    result = []
    for r in res:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        certainty = float(r["certainty"]["value"])
        dataset = r["dataset2_n"]["value"]
        path = r["path"]["value"]
        result.append([(table1, table2), certainty, dataset, path])

    result = get_top_k_tables(result)[:k]
    table = list(map(lambda x: x[1], result))
    scores = list(map(lambda x: x[2], result))
    dataset = list(map(lambda x: x[3], result))
    path = list(map(lambda x: x[4], result))
    return pd.DataFrame({'Dataset': dataset, 'Recommended_table': table,
                         'Score': scores, 'Path_to_table': path})


def show_graph_info(config, show_query):
    query1 = PREFIXES + """
    SELECT (COUNT(?Dataset) as ?Datasets)
    WHERE
    {
        ?Dataset    rdf:type kglids:Dataset     .
    }
    """
    query2 = PREFIXES + """
    SELECT  (COUNT(?Table) as ?Tables)
    WHERE
    {
        ?Table      rdf:type        kglids:Table    ;
                    kglids:isPartOf ?Dataset        .
        ?Dataset    rdf:type        kglids:Dataset  . 
    } 
    """
    query3 = PREFIXES + """
    SELECT (COUNT(?Pipeline) as ?Pipelines)
    WHERE
    {
        ?Pipeline   rdf:type    kglids:Pipeline ;
                    kglids:isPartOf ?Dataset    .
        ?Dataset    rdf:type    kglids:Dataset  .
    }
    """
    query4 = PREFIXES + """
    SELECT  (COUNT(?Column) as ?Columns)
    WHERE
    {
        ?Column     rdf:type        kglids:Column   ;
                    kglids:isPartOf ?Table          .
        ?Table      rdf:type        kglids:Table    . 
    } 
    """
    if show_query:
        print(query1, '\n', query3, '\n', query2, '\n', query4)
    dataset = query_graphdb(config, query1)
    tables = query_graphdb(config, query2)
    pipelines = query_graphdb(config, query3)
    columns = query_graphdb(config, query4)
    return pd.concat([dataset, pipelines, tables, columns], axis=1)


def get_table_path(config, dataset, table):
    query = PREFIXES + """
    SELECT ?table_path
	WHERE
	{
      ?dataset	schema:name				"%s"	;
      			rdf:type				kglids:Dataset	.
      ?table	schema:name				"%s"	;
      			rdf:type				kglids:Table	;
      			data:hasFilePath		?table_path		.			
    }
    """ % (dataset, table)

    res = query_graphdb(config, query)["results"]["bindings"][0]
    return res["table_path"]["value"]


def get_table_info(config, dataset, table, show_query):
    query = PREFIXES + """
    SELECT (max(?rows) as ?number_of_rows) (COUNT(?col) as ?number_of_columns) 
	WHERE
	{
      ?dataset	schema:name				"%s"	;
      			rdf:type				kglids:Dataset	.
      ?table	schema:name				"%s"	;
      			rdf:type				kglids:Table	;
      			data:hasFilePath		?table_path		.
      ?col		kglids:isPartOf			?table			;	
                rdf:type				kglids:Column	;
      			data:hasTotalValueCount	?rows			.   			
    } 
    """ % (dataset, table)
    if show_query:
        print(query)
    res = query_graphdb(config, query)["results"]["bindings"][0]
    rows = res["number_of_rows"]["value"]
    columns = res["number_of_columns"]["value"]

    return pd.DataFrame(
        {'Dataset': [dataset], 'Table': [table], 'Path_to_table':
            [get_table_path(config, dataset, table)], 'Number_of_columns': [columns],
         'Number_of_rows': [rows]})


def _create_tables_df_row(results):
    return {
        'Dataset': results['dataset_name']['value'],
        'Table': results['name']['value'],
        'Number_of_columns': float(results['number_of_columns']['value']),
        'Number_of_rows': float(results['number_of_rows']['value']),
        'Path_to_table': results['path']['value']
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
    res = query_graphdb(config, query, return_type='json')

    for result in res:
        yield _create_tables_df_row(result)


def _get_iri(config, dataset_name: str, table_name: str = None, show_query: bool = False):
    if table_name is None:
        query = PREFIXES + \
                '\nselect ?id' \
                '\nwhere {' \
                '\n?id a kglids:Dataset.' \
                '\n?id rdfs:label %s }' % dataset_name
    else:
        query = PREFIXES + \
                '\nselect ?id where{' \
                '\n?id a kglids:Table.' \
                '\n?id rdfs:label %s.' \
                '\n?id kglids:isPartOf ?dataset.' \
                '\n?dataset rdfs:label %s.' \
                '\n?dataset a kglids:Dataset.}' % (table_name, dataset_name)
    if show_query:
        print(query)
    results = query_graphdb(config, query, return_type='json')
    bindings = results
    if not bindings:
        return None
    return str(bindings[0]['id']['value'])


def get_iri_of_table(config, dataset_name: str, table_name: str, show_query: bool = False):
    dataset_label = generate_label(dataset_name, 'en')
    table_label = generate_label(table_name, 'en')
    return _get_iri(config, dataset_label, table_label, show_query)


def generate_label(col_name: str, lan: str) -> Label:
    if '.csv' in col_name:
        col_name = re.sub('.csv', '', col_name)
    col_name = re.sub('[^0-9a-zA-Z]+', ' ', col_name)
    text = " ".join(camelsplit(col_name.strip()))
    text = re.sub(r'\s+', ' ', text.strip())
    return Label(text.lower(), lan)


def _create_path_row(result, hops):
    data = {'starting_column': result['c1name']['value'],
            'starting_table': result['t1name']['value'],
            'starting_table_path': result['t1path']['value'],
            'starting_dataset': result['d1name']['value']}

    intermediate = {}
    for i in range(2, hops + 1):
        intermediate.update({'intermediate_column_land_in' + str(i): result['c' + str(i) + 'name']['value'],
                             'intermediate_table' + str(i): result['t' + str(i) + 'name']['value'],
                             'intermediate_table_path' + str(i): result['t' + str(i) + 'path']['value'],
                             'intermediate_column_take_off' + str(i): result['cc' + str(i) + 'name']['value'],
                             'intermediate_dataset' + str(i): result['d' + str(i) + 'name']['value']})
    data.update(intermediate)

    data.update({'target_column': result['c' + str(hops + 1) + 'name']['value'],
                 'target_table': result['t' + str(hops + 1) + 'name']['value'],
                 'target_table_path': result['t' + str(hops + 1) + 'path']['value'],
                 'target_dataset': result['d' + str(hops + 1) + 'name']['value']})
    return data


def get_path_between(config, start_iri: str, target_iri: str, predicate: str, hops: int,
                     show_query: bool = False):
    def _generate_starting_nodes() -> str:
        return '\n    ?c1 schema:name ?c1name.' \
               '\n    ?c1 kglids:isPartOf ?t1.' \
               '\n    ?t1 schema:name ?t1name.' \
               '\n    ?t1 data:hasFilePath ?t1path.' \
               '\n    ?t1 kglids:isPartOf ?d1.' \
               '\n    ?d1 schema:name ?d1name.'

    def _generate_intermediate_nodes(h: int) -> str:
        inters = ''
        for i in range(2, h + 1):
            inter = '\n    ?c' + str(i) + ' a kglids:Column.' \
                                          '\n    ?c' + str(i) + ' schema:name ?c' + str(i) + 'name.' \
                                                                                             '\n    ?c' + str(
                i) + ' kglids:isPartOf ?t' + str(i) + '.' \
                                                      '\n    ?t' + str(i) + ' schema:name ?t' + str(i) + 'name.' \
                                                                                                         '\n    ?t' + str(
                i) + ' data:hasFilePath ?t' + str(i) + 'path.' \
                                                       '\n    ?cc' + str(i) + ' kglids:isPartOf ?t' + str(i) + '.' \
                                                                                                               '\n    ?cc' + str(
                i) + ' schema:name ?cc' + str(i) + 'name.' \
                                                   '\n    ?t' + str(i) + ' kglids:isPartOf ?d' + str(i) + '.' \
                                                                                                          '\n    ?d' + str(
                i) + ' schema:name ?d' + str(i) + 'name.'
            inters += inter
        return inters

    def _generate_target_nodes(h: int) -> str:
        return '\n    ?c' + str(h + 1) + ' schema:name ?c' + str(h + 1) + 'name.' \
                                                                          '\n    ?c' + str(
            h + 1) + ' kglids:isPartOf ?t' + str(h + 1) + '.' \
                                                          '\n    ?t' + str(h + 1) + ' schema:name ?t' + str(
            h + 1) + 'name.' \
                     '\n    ?t' + str(h + 1) + ' data:hasFilePath ?t' + str(h + 1) + 'path.' \
                                                                                     '\n    ?t' + str(
            h + 1) + ' kglids:isPartOf ?d' + str(h + 1) + '.' \
                                                          '\n    ?d' + str(h + 1) + ' schema:name ?d' + str(
            h + 1) + 'name.'

    def _generate_relationships(h: int, pred: str) -> str:
        relations = '\n    << ?c1 ' + pred + ' ?c2 >> data:withCertainty	?certainty1'
        for i in range(2, h + 1):
            relation = f'\n     << ?cc{i} {pred} ?c{i+1}>> data:withCertainty	?certainty{i}.'
            relations += relation
        return relations

    def _generate_select(h: int) -> str:
        selects = '\n ?c1name ?t1name ?t1path ?d1name'
        for i in range(2, h + 1):
            selects += '\n ?c' + str(i) + 'name ?t' + str(i) + 'name ?t' + str(i) + 'path' \
                                                                                    ' ?cc' + str(i) + 'name ?d' + str(
                i) + 'name'
        selects += '\n ?c' + str(h + 1) + 'name ?t' + str(h + 1) + 'name ?t' + str(h + 1) + 'path ?d' + str(
            h + 1) + 'name'
        return selects

    starting_nodes = _generate_starting_nodes()
    intermediate_nodes = _generate_intermediate_nodes(hops)
    target_nodes = _generate_target_nodes(hops)
    relationships = _generate_relationships(hops, predicate)
    select = _generate_select(hops)
    all_nodes = starting_nodes + intermediate_nodes + target_nodes

    query = PREFIXES + \
            '\n select' + select + \
            '\nwhere {' + \
            all_nodes + relationships + \
            '\n values ?t1 {' + start_iri + '}' + \
            '\n values ?t' + str(hops + 1) + ' {' + target_iri + '}}'

    if show_query:
        print(query)
    results = query_graphdb(config, query, return_type='json')
    bindings = results
    if not bindings:
        return []
    for result in bindings:
        yield _create_path_row(result, hops)


def generate_component_id(dataset_name: str, table_name: str = '', column_name: str = ''):
    return zlib.crc32(bytes(dataset_name + table_name + column_name, 'utf-8'))


def generate_graphviz(df: pd.DataFrame, predicate: str):
    def parse_starting_or_target_nodes(dot, row, column_ids: list, table_ids: list, dataset_ids: list,
                                       start: bool) -> str:
        relation_name = 'partOf'
        if start:
            dataset_name = row[0]
            table_name = row[1]
            column_name = row[3]
            color = 'lightblue2'
        else:
            dataset_name = row[-4]
            table_name = row[-3]
            column_name = row[-1]
            color = 'darkorange3'
        dataset_id = str(generate_component_id(dataset_name))
        table_id = str(generate_component_id(dataset_name, table_name))
        column_id = str(generate_component_id(dataset_name, table_name, column_name))
        if column_id in column_ids:
            return column_id
        dot.node(column_id, column_name, style='filled', fillcolor=color)
        column_ids.append(column_id)
        if table_id in table_ids:
            dot.edge(column_id, table_id, relation_name)
            return column_id
        dot.node(table_id, table_name, style='filled', fillcolor=color)
        table_ids.append(table_id)
        if dataset_id in dataset_ids:
            dot.edge(column_id, table_id, relation_name)
            dot.edge(table_id, dataset_id, relation_name)
            return column_id
        dot.node(dataset_id, dataset_name, style='filled', fillcolor=color)
        dataset_ids.append(dataset_id)
        dot.edge(column_id, table_id, relation_name)
        dot.edge(table_id, dataset_id, relation_name)
        return column_id

    def parse_intermediate_nodes(dot, row, column_ids: list, table_ids: list, dataset_ids: list) -> list:
        ids = []
        relation_name = 'partOf'
        for i in range(4, len(row) - 4, 5):
            dataset_name = row[i]
            table_name = row[i + 1]
            land_in_column_name = row[i + 2]
            take_off_column_name = row[i + 4]
            dataset_id = str(generate_component_id(dataset_name))
            table_id = str(generate_component_id(dataset_name, table_name))
            land_in_column_id = str(generate_component_id(dataset_name, table_name, land_in_column_name))
            take_off_column_id = str(generate_component_id(dataset_name, table_name, take_off_column_name))
            ids.extend([land_in_column_id, take_off_column_id])

            land_in_column_exist = False
            take_off_column_exist = False
            if land_in_column_id in column_ids:
                land_in_column_exist = True
            dot.node(land_in_column_id, land_in_column_name)
            column_ids.append(land_in_column_id)

            if take_off_column_id in column_ids:
                take_off_column_exist = True

            if land_in_column_exist and take_off_column_exist:
                continue
            dot.node(take_off_column_id, take_off_column_name)
            column_ids.append(take_off_column_id)

            if table_id in table_ids:
                if land_in_column_id == take_off_column_id:
                    dot.edge(land_in_column_id, table_id, relation_name)
                else:
                    dot.edge(land_in_column_id, table_id, relation_name)
                    dot.edge(take_off_column_id, table_id, relation_name)
                continue

            dot.node(table_id, table_name)
            table_ids.append(table_id)
            if dataset_id in dataset_ids:
                if land_in_column_id == take_off_column_id:
                    dot.edge(land_in_column_id, table_id, relation_name)
                else:
                    dot.edge(land_in_column_id, table_id, relation_name)
                    dot.edge(take_off_column_id, table_id, relation_name)
                dot.edge(table_id, dataset_id, relation_name)
                continue
            dot.node(dataset_id, dataset_name)
            dataset_ids.append(dataset_id)
            if land_in_column_id == take_off_column_id:
                dot.edge(land_in_column_id, table_id, relation_name)
            else:
                dot.edge(land_in_column_id, table_id, relation_name)
                dot.edge(take_off_column_id, table_id, relation_name)
            dot.edge(table_id, dataset_id, relation_name)
        return ids

    def establish_relationships(dot, row_ids: list, relationships: list):

        for j in range(0, len(row_ids) - 1, 2):
            pair = (row_ids[j], row_ids[j + 1])
            if pair[0] == pair[1]:
                continue
            if not pair in relationships:
                relationships.append(pair)
                dot.edge(pair[0], pair[1], 'similar', dir='none')

    col_ids = []
    tab_ids = []
    data_ids = []
    relations = []
    dot_graph = Digraph(strict=True)
    for i in range(len(df)):
        r = df.iloc[i]
        row_col_ids = []
        starting_column_id = parse_starting_or_target_nodes(dot_graph, r, col_ids, tab_ids, data_ids, True)
        intermediate_col_ids = parse_intermediate_nodes(dot_graph, r, col_ids, tab_ids, data_ids)
        target_col_id = parse_starting_or_target_nodes(dot_graph, r, col_ids, tab_ids, data_ids, False)

        row_col_ids.append(starting_column_id)
        row_col_ids.extend(intermediate_col_ids)
        row_col_ids.append(target_col_id)

        establish_relationships(dot_graph, row_col_ids, relations)
    dot_graph.attr(label='Paths between starting nodes in blue and target nodes in orange')

    return dot_graph


def get_path_between_tables(config, source_table_info, target_table_info, hops, relation, show_query):
    source_table_name = source_table_info["Table"]
    source_dataset_name = source_table_info["Dataset"]
    target_dataset_name = target_table_info["Dataset"]
    if 'Recommended_table' in target_table_info.keys():
        target_table_name = target_table_info["Recommended_table"]
    else:
        target_table_name = target_table_info["Table"]

    starting_table_iri = get_iri_of_table(config,
                                          dataset_name=generate_label(source_dataset_name, 'en').get_text(),
                                          table_name=generate_label(source_table_name, 'en').get_text())
    target_table_iri = get_iri_of_table(config,
                                        dataset_name=generate_label(target_dataset_name, 'en').get_text(),
                                        table_name=generate_label(target_table_name, 'en').get_text())

    if starting_table_iri is None:
        raise ValueError(str(source_table_info) + ' does not exist')
    if target_table_iri is None:
        raise ValueError(str(target_table_info) + ' does not exist')

    data = get_path_between(config, '<' + starting_table_iri + '>', '<' + target_table_iri + '>',
                            relation, hops, show_query)

    path_row = ['starting_dataset', 'starting_table', 'starting_table_path', 'starting_column']
    for i in range(2, hops + 1):
        intermediate = ['intermediate_dataset' + str(i), 'intermediate_table' + str(i),
                        'intermediate_column_land_in' + str(i), 'intermediate_table_path' + str(i),
                        'intermediate_column_take_off' + str(i)]
        path_row.extend(intermediate)
    path_row.extend(['target_dataset', 'target_table', 'target_table_path', 'target_column'])
    df = pd.DataFrame(list(data), columns=path_row)
    dot = generate_graphviz(df, relation)
    return dot


def get_top_scoring_ml_model(config, dataset, show_query):
    query = """
    PREFIX kglids: <http://kglids.org/ontology/>
    SELECT (count(?x) as ?count)
    WHERE
    {
        ?x rdf:type kglids:Pipeline .
    }
    """
    return query_graphdb(config, query)


def get_pipelines_info(config, author, show_query):
    if author != '':
        author = "FILTER (?Author = '{}')   .".format(author)

    query = PREFIXES + """
    SELECT ?Pipeline ?Dataset ?Author ?Written_on ?Number_of_votes ?Score
    WHERE
    {
        ?pipeline_id    rdf:type                kglids:Pipeline     ;
                        pipeline:hasVotes       ?Number_of_votes    ;
                        rdfs:label              ?Pipeline           ;
                        pipeline:isWrittenOn    ?Written_on         ;
                        pipeline:isWrittenBy    ?Author             ;
                        pipeline:hasScore       ?Score              ;
                        kglids:isPartOf         ?Dataset_id         .
        ?Dataset_id     schema:name             ?Dataset            .
        %s
    } ORDER BY DESC(?Number_of_votes) 
    """ % author
    if show_query:
        print(query)

    return query_graphdb(config, query)


def get_most_recent_pipeline(config, dataset, show_query):
    if dataset != '':
        dataset = "FILTER (?Dataset = '{}')     .".format(dataset)

    query = PREFIXES + """
    SELECT ?Pipeline ?Dataset ?Author ?Written_on ?Number_of_votes ?Score
    WHERE
    {
        ?pipeline_id    rdf:type                kglids:Pipeline     ;
                        pipeline:hasVotes       ?Number_of_votes    ;
                        rdfs:label              ?Pipeline           ;
                        pipeline:isWrittenOn    ?Written_on         ;
                        pipeline:isWrittenBy    ?Author             ;
                        pipeline:hasScore       ?Score              ;
                        kglids:isPartOf         ?Dataset_id         .
        ?Dataset_id     schema:name             ?Dataset            .
        %s                          
    } ORDER BY DESC(?Written_on) LIMIT 1
    """ % dataset
    if show_query:
        print(query)
    return query_graphdb(config, query)


def get_top_k_scoring_pipelines_for_dataset(config, dataset, k, show_query):
    if k is not None:
        k = 'LIMIT ' + str(k)
    if k is None:
        k = ''
    if dataset != '':
        dataset = "FILTER (?Dataset = '{}')     .".format(dataset)
    query = PREFIXES + """
    SELECT ?Pipeline ?Dataset ?Author ?Written_on ?Number_of_votes ?Score
    WHERE
    {
        ?pipeline_id    rdf:type                kglids:Pipeline     ;
                        pipeline:hasVotes       ?Number_of_votes    ;
                        rdfs:label              ?Pipeline           ;
                        pipeline:isWrittenOn    ?Written_on         ;
                        pipeline:isWrittenBy    ?Author             ;
                        pipeline:hasScore       ?Score              ;
                        kglids:isPartOf         ?Dataset_id         .
        ?Dataset_id     schema:name             ?Dataset            .
        %s                
    } ORDER BY DESC(?Score) %s
    """ % (dataset, k)
    if show_query:
        print(query)
    return query_graphdb(config, query)


CLASSIFIERS = {'RandomForestClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier>',
               'SVC': '<http://kglids.org/resource/library/sklearn/svm/SVC>',
               'KNeighborsClassifier': '<http:/kglids.org/resource/library/sklearn/neighbors/KNeighborsClassifier>',
               'GradientBoostingClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingClassifier>',
               'LogisticRegression': '<http://kglids.org/resource/library/sklearn/linear_model/LogisticRegression>',
               'DecisionTreeClassifier': '<http://kglids.org/resource/library/sklearn/tree/DecisionTreeClassifier>',
               'AdaBoostClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/AdaBoostClassifier>',
               'SGDClassifier': '<http://kglids.org/resource/library/sklearn/linear_model/SGDClassifier>',
               'MLPClassifier': '<http://kglids.org/resource/library/sklearn/neural_network/MLPClassifier>',
               'XGBClassifier': '<http://kglids.org/resource/library/xgboost/XGBClassifier>',
               'VotingClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/VotingClassifier>',
               'PassiveAggressiveClassifier': '<http://kglids.org/resource/library/sklearn/linear_model/PassiveAggressiveClassifier>',
               'BaggingClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/BaggingClassifier>',
               'RidgeClassifier': '<http://kglids.org/resource/library/sklearn/linear_model/RidgeClassifier>',
               'RadiusNeighborsClassifier': '<http://kglids.org/resource/library/sklearn/neighbors/RadiusNeighborsClassifier>',
               'ExtraTreesClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/ExtraTreesClassifier>',
               'TFDistilBertForSequenceClassification': '<http://kglids.org/resource/library/transformers/TFDistilBertForSequenceClassification>'}


def search_classifier(config, dataset, show_query):
    sub_graph_query = """
    """
    for classifier, classifier_url in CLASSIFIERS.items():
        if classifier != 'RandomForestClassifier':
            query = """
            UNION
                     {
                        ?Statement_number    pipeline:callsClass %s.
                        BIND('%s' as ?Classifier)
                     }
            """ % (classifier_url, classifier)
            sub_graph_query = sub_graph_query + query

    filter_for_dataset = ''
    if dataset != '':
        filter_for_dataset = '?Dataset_id    schema:name "{}".'.format(dataset)

    query = PREFIXES + """
    SELECT DISTINCT ?Dataset ?Pipeline ?Classifier ?Score
    WHERE 
    {
        ?Dataset_id     rdf:type          kglids:Dataset ;
                        schema:name       ?Dataset       .
        ?Pipeline_id    kglids:isPartOf   ?Dataset_id    ;
                        rdfs:label        ?Pipeline      ;
                        pipeline:hasScore ?Score         .
       graph ?Pipeline_id 
         {
            ?x pipeline:callsClass ?y
             {
                ?Statement_number    pipeline:callsClass <http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier>  .
                BIND('RandomForestClassifier' as ?Classifier)
             }""" + sub_graph_query + """
                          
         }
    %s
    } ORDER BY DESC(?Score) """ % filter_for_dataset

    if show_query:
        print(query)

    return query_graphdb(config, query)


def get_hyperparameters(config, pipeline, classifier, show_query):
    classifier_url = CLASSIFIERS.get(classifier)
    parameter_heading = '?{}_hyperparameter'.format(classifier)
    query = PREFIXES + """
    
    SELECT DISTINCT %s ?Value
        WHERE
        {
            ?Pipeline_id    rdfs:label        '%s'     ;
                            pipeline:hasScore ?Score         .
           graph ?Pipeline_id
             {
                 ?Statement_number    pipeline:callsClass   %s .
                 << ?Statement_number pipeline:hasParameter %s >> pipeline:withParameterValue ?Value  .
             }
        } ORDER BY DESC(?Score)""" % (parameter_heading, pipeline, classifier_url, parameter_heading)

    if show_query:
        print(query)

    df = query_graphdb(config, query)
    if np.shape(df)[0] == 0:
        return 'Using default configurations'
    else:
        return df


def get_library_usage(config, dataset, k, show_query):
    if dataset != '':
        dataset = '?Dataset    schema:name        "{}"        .\n\t\t' \
                  '?Pipeline   kglids:isPartOf    ?Dataset  .'.format(dataset)
    query = PREFIXES + """
    SELECT ?Library (COUNT(distinct ?Pipeline) as ?Usage)
    WHERE
    {
        %s
        ?Pipeline   rdf:type    kglids:Pipeline                 .
        GRAPH ?Pipeline
        {
            ?Statement pipeline:callsClass ?l                 .
            BIND(STRAFTER(str(?l), str(lib:)) as ?l1)      .
            BIND(STRBEFORE(str(?l1), str('/')) as ?Library)     .
        }
        FILTER (?Library != "")              .
        FILTER (?Library != "builtin")       .         
    } GROUP BY ?Library ORDER BY DESC(?Usage)
    """ % dataset
    if show_query:
        print(query)
    df = query_graphdb(config, query)
    df['Usage (in %)'] = list(map(lambda x: x * 100, [int(i) / sum(df['Usage'].
                                                                   tolist()) for i in (df['Usage'].tolist())]))
    if len(df) == 0:
        print("No library found")
        return
    
    df = df.head(k)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(df)), df['Usage'].tolist(), color='mediumseagreen')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Library'].tolist(), rotation=20)
    ax.set_ylabel('Number of Pipelines')
    for i, usage in enumerate(df['Usage']):
        ax.text(i-0.35, usage + int(0.05 * usage), str(usage), color='k', fontweight='bold')
    ax.set_ylim(0, df['Usage'].max() + int(0.1 * df['Usage'].max()))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('library_usage_stats.pdf')
    plt.show()



def get_top_used_libraries(config, task, show_query):
    if task == 'classification':
        task = 'classifi'
    elif task == 'clustering':
        task = 'cluster'
    elif task == 'visualization':
        task = 'plot'
    else:
        task = 'regress'
    query = PREFIXES + """
    SELECT DISTINCT ?Library ?Module ?Pipeline ?Dataset 
    WHERE
    {
        ?Pipeline_id    rdf:type                    kglids:Pipeline         ;
                        rdfs:label                  ?Pipeline               ;
                        kglids:isPartOf             ?dataset_id             .
        
        ?dataset_id     schema:name                 ?Dataset                .
        
        GRAPH ?Pipeline_id
        {
            ?statement  pipeline:callsFunction    ?l                         .
            BIND(STRAFTER(str(?l), str(lib:)) as ?l1)                       .
            BIND(STRBEFORE(str(?l1), str('/')) as ?Library)                 .  
            BIND (REPLACE(STR(?l), "^.*/([^/]*)/([^/]*)$", "$1") as ?Module)        .    
        }    
    
        FILTER(regex(str(?l), "%s", "i"))             
    }""" % task
    if show_query:
        print(query)

    return query_graphdb(config, query)


def get_pipelines_calling_libraries(config, components, show_query):
    sub_query = ''
    for i in range(len(components)):
        # check if class or function (classes start with an upper case letter (heuristic))
        if components[i].split('.')[-1][0].isupper(): 
            predicate = 'pipeline:callsClass'
        else:
            predicate = 'pipeline:callsFunction'
        sub_query = sub_query + \
                    '?Statement_{}   {}   <http://kglids.org/resource/library/{}> .\n        ' \
                        .format(i + 1, predicate, components[i].replace('.', '/'))

    query = PREFIXES + """
    SELECT DISTINCT ?Pipeline ?Dataset ?Author ?Score ?Number_of_votes
    WHERE 
    {
    
        ?Pipeline_id    rdf:type                kglids:Pipeline     ;
                        pipeline:hasVotes       ?Number_of_votes    ;
                        rdfs:label              ?Pipeline           ;
                        pipeline:isWrittenOn    ?Written_on         ;
                        pipeline:isWrittenBy    ?Author             ;
                        pipeline:hasScore       ?Score              ;
                        kglids:isPartOf         ?Dataset_id         ;
        
        graph ?Pipeline_id
        {
        %s
        }  
        ?Dataset_id     schema:name             ?Dataset            . 
    } ORDER BY DESC(?Score)""" % sub_query
    if show_query:
        print(query)

    return query_graphdb(config, query)


def get_pipelines_for_deep_learning(config, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Pipeline ?Dataset ?Author ?Written_on ?Score ?Number_of_votes
    WHERE 
    {
    ?Pipeline_id    rdf:type                kglids:Pipeline     ;
                    pipeline:hasVotes       ?Number_of_votes    ;
                    rdfs:label              ?Pipeline           ;
                    pipeline:isWrittenOn    ?Written_on         ;
                    pipeline:isWrittenBy    ?Author             ;
                    pipeline:hasScore       ?Score              ;
                    pipeline:hasTag         ?Tag                ;
                    kglids:isPartOf         ?Dataset_id         .
    
    ?Dataset_id     schema:name             ?Dataset            . 
    
    FILTER(regex(?Tag, "deep learning", "i")) 
    } ORDER BY DESC(?Score)"""
    if show_query:
        print(query)

    return query_graphdb(config, query)


def recommend_transformations(config, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Transformation ?Pipeline (?Table_id as ?Table) ?Dataset 
    WHERE
    {
    ?Pipeline_id    rdf:type                kglids:Pipeline     ;
                    rdfs:label              ?Pipeline           ;
                    kglids:isPartOf         ?Dataset_id         .
    
    ?Dataset_id     schema:name             ?Dataset            . 
    
    graph ?Pipeline_id
    {   
        ?s          pipeline:callsClass    ?l                 . 
        BIND(STRAFTER(str(?l), str(lib:)) as ?Transformation)   . 
        ?Table_id   rdf:type                 kglids:Table       . 
                            
    }
    FILTER(regex(str(?l), "preprocessing", "i"))
    } ORDER BY DESC(?Score) """

    if show_query:
        print(query)

    df = query_graphdb(config, query)
    df['Table'] = df['Table'].apply(lambda x: x.rsplit('/', 1)[-1])
    df['Transformation'] = df['Transformation'].apply(lambda x: x.replace('/', '.'))
    return df


def get_pipelines_by_tags(config, tag, show_query):
    if tag != '':
        tag = 'FILTER(regex(?Tag, "{}", "i"))'.format(tag)
    query = PREFIXES + """
    SELECT DISTINCT ?Tag (COUNT (?Pipeline_id) AS ?Number_of_pipelines) 
    WHERE
    {
    ?Pipeline_id    rdf:type                kglids:Pipeline     ;
                    pipeline:hasTag         ?Tag                .
    %s
    } GROUP BY ?Tag ORDER BY DESC(?Number_of_pipelines)
    """ % tag
    if show_query:
        print(query)
    return query_graphdb(config, query)


def plot_top_k_classifiers(config, k, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Module (COUNT (?Module) as ?Usage)
    WHERE
    {
    ?Pipeline_id    rdf:type                    kglids:Pipeline         ;
                    rdfs:label                  ?Pipeline               ;
                    kglids:isPartOf             ?dataset_id             .
    
    ?dataset_id     schema:name                 ?Dataset                .
    
    GRAPH ?Pipeline_id
    {
        ?statement  pipeline:callsFunction    ?l                         .
        BIND(STRAFTER(str(?l), str(lib:)) as ?l1)                       .
        BIND(STRBEFORE(str(?l1), str('/')) as ?Library)                 .  
        BIND (REPLACE(STR(?l), "^.*/([^/]*)/([^/]*)$", "$1") as ?Module)                  .  
    }    
    
    FILTER(regex(str(?l), "classifier", "i"))
    FILTER (!regex(str(?l), "report", "i"))
    FILTER (!regex(str(?l), "ROCAUC", "i"))  
    FILTER (!regex(str(?l), "threshold", "i"))  
    } GROUP BY ?Module ORDER BY DESC(?Usage)"""

    if show_query:
        print(query)
    df = query_graphdb(config, query)
    df['Usage (in %)'] = list(map(lambda x: x * 100, [int(i) / sum(df['Usage'].
                                                                   tolist()) for i in (df['Usage'].tolist())]))

    df['Classifier'] = df['Module'].apply(lambda x: x.rsplit('/', 1)[-1])

    if len(df) == 0:
        print("No classifier found")
        return
    if len(df) < k:
        print('Maximum {} classifier(s) were found'.format(len(df)))
        print('Showing top-{} classifiers'.format(len(df)))
        k = len(df)
    df = df.head(k)
    plt.rcParams['figure.figsize'] = 10, 5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set_theme(style='darkgrid')
    ax = sns.barplot(x="Classifier", y="Usage (in %)", data=df, palette='viridis')
    ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=40)


def plot_top_k_regressors(config, k , show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Module (COUNT (?Module) as ?Usage)
    WHERE
    {
    ?Pipeline_id    rdf:type                    kglids:Pipeline         ;
                    rdfs:label                  ?Pipeline               ;
                    kglids:isPartOf             ?dataset_id             .

    ?dataset_id     schema:name                 ?Dataset                .

    GRAPH ?Pipeline_id
    {
        ?statement  pipeline:callsFunction    ?l                         .
        BIND(STRAFTER(str(?l), str(lib:)) as ?l1)                       .
        BIND(STRBEFORE(str(?l1), str('/')) as ?Library)                 .  
        BIND (REPLACE(STR(?l), "^.*/([^/]*)/([^/]*)$", "$1") as ?Module)                    .  
    }    

    FILTER(regex(str(?l), "regres", "i"))
    } GROUP BY ?Module ORDER BY DESC(?Usage)"""

    if show_query:
        print(query)

    df = query_graphdb(config, query)
    df['Usage (in %)'] = list(map(lambda x: x * 100, [int(i) / sum(df['Usage'].
                                                                    tolist()) for i in (df['Usage'].tolist())]))

    df['Regressor'] = df['Module'].apply(lambda x: x.rsplit('/', 1)[-1])

    if len(df) == 0:
        print("No classifier found")
        return
    if len(df) < k:
        print('Maximum {} regressor(s) were found'.format(len(df)))
        print('Showing top-{} regressors'.format(len(df)))
        k = len(df)
    df = df.head(k)
    plt.rcParams['figure.figsize'] = 10, 5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set_theme(style='darkgrid')
    ax = sns.barplot(x="Regressor", y="Usage (in %)", data=df, palette='viridis')
    ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=40)