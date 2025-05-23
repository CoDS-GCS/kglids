import os
import spacy
import fasttext
import flask
from flask.globals import request
import pandas as pd
import psycopg
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp

from kglids_config import KGLiDSConfig
from server_utils import (query_graph, create_evaluation_embedding_dbs, add_has_eda_ops_column_to_embedding_db,
                          upload_dataset_subgraph_to_evaluation_graph, upload_pipeline_subgraph_to_evaluation_graph)
from storage_utils.graphdb_utils import create_graphdb_repo
from kg_governor.data_profiling.fine_grained_type_detector import FineGrainedColumnTypeDetector
from kg_governor.data_profiling.profile_creators.profile_creator import ProfileCreator
from kg_governor.data_profiling.model.column_profile import ColumnProfile
from kg_governor.data_profiling.model.table import Table

app = flask.Flask(__name__)

fasttext_model_300 = fasttext.load_model(os.path.join(KGLiDSConfig.base_dir, 'storage/embeddings/cc.en.300.bin'))
fasttext_model_50 = fasttext.load_model(os.path.join(KGLiDSConfig.base_dir, 'storage/embeddings/cc.en.50.bin'))
ner_model = spacy.load('en_core_web_sm')

def initialize_autoeda():
    graphdb_endpoint = 'http://localhost:7200/repositories/kaggle_eda'
    embedding_db_name = 'kaggle_eda_column_embeddings'
    add_has_eda_ops_column_to_embedding_db(embedding_db_name, graphdb_endpoint)


def profile_column(args):
    column, query_table_name = args
    # profiles each column by analyzing its fine-grained type and generating the embedding
    # profile the column and generate its embeddings
    column = pd.to_numeric(column, errors='ignore')
    column = column.convert_dtypes()
    column = column.astype(str) if column.dtype == object else column

    column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column, fasttext_model_50, ner_model)
    column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, Table('query',
                                                                                           query_table_name,
                                                                                           'query'),
                                                                fasttext_model_50)
    column_profile: ColumnProfile = column_profile_creator.create_profile()
    if column_profile.get_embedding():
        content_embedding = column_profile.get_embedding()
    else:
        # boolean columns
        content_embedding = [column_profile.get_true_ratio()] * 300

    sanitized_name = column.name.replace('\n', ' ').replace('_', ' ').strip()
    label_embedding = fasttext_model_300.get_sentence_vector(sanitized_name).tolist()

    content_label_embedding = content_embedding + label_embedding
    column_info = {'column_id': column_profile.get_column_id(),
                   'column_name': column_profile.get_column_name(),
                   'data_type': column_profile.get_data_type(),
                   'unique_values_count': column_profile.get_distinct_values_count(),
                   'missing_values_count': column_profile.get_missing_values_count(),
                   'dataset_name': column_profile.get_dataset_name(),
                   'table_name': column_profile.get_table_name(),
                   'content_embedding': str(content_embedding),
                   'label_embedding': str(label_embedding),
                   'content_label_embedding': str(content_label_embedding)}
    return column_info


@app.route('/profile_query_table', methods=['POST'])
def profile_query_table():
    embedding_db_name = request.args.get('embedding_db_name')
    query_table_name = request.args.get('query_table_name')
    file = request.files['query_table']
    df = pd.read_csv(file)

    print(datetime.now(), 'Received CSV file. Profiling and storing embeddings...')
    args = [(df[column_name], query_table_name) for column_name in df.columns]
    pool = mp.Pool()
    column_info = list(tqdm(pool.imap_unordered(profile_column, args), total=len(args)))

    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    # remove previous query table if exists
    cursor.execute(f"DELETE FROM {embedding_db_name} WHERE dataset_name = 'query' AND table_name = '{query_table_name}'")
    # add new query table to embedding store
    insert_query = f'''INSERT INTO {embedding_db_name} (id, name, data_type, dataset_name, table_name, has_eda_ops, 
                       content_embedding, label_embedding, content_label_embedding) 
                       VALUES (%s, %s, %s, %s, %s, FALSE, %s, %s, %s);'''
    insert_data = [(i['column_id'], i['column_name'], i['data_type'], 'query', query_table_name,
                    i['content_embedding'], i['label_embedding'], i['content_label_embedding'])
                   for i in column_info]
    cursor.executemany(insert_query, insert_data)
    cursor.close()

    return_data = [{k: i[k] for k in ['column_name', 'data_type', 'unique_values_count', 'missing_values_count']}
                   for i in column_info]
    return flask.jsonify(return_data)


@app.route('/find_similar_columns', methods=['POST'])
def find_similar_columns():

    column_name = request.get_json().get('main_column_name', '')
    embedding_db_name = request.get_json().get('embedding_db_name', '')
    criteria = request.get_json().get('criteria', '') # has to be 'content', 'label', or 'content_label'
    query_table_name = request.get_json().get('query_table_name')
    n = request.get_json().get('n', 3)

    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres')
    cursor = conn.cursor()
    column_fetch_query = f"""SELECT data_type, content_embedding, label_embedding, content_label_embedding
                            FROM {embedding_db_name} 
                            WHERE dataset_name = 'query' AND table_name = '{query_table_name}' and name = '{column_name.replace("'", "''")}'"""
    results = cursor.execute(column_fetch_query)
    data_type, content_embedding, label_embedding, content_label_embedding = results.fetchone()

    criteria_name_and_embedding = {'content': ('content_embedding', content_embedding),
                                   'label': ('label_embedding', label_embedding),
                                   'content_label': ('content_label_embedding', content_label_embedding)}
    filter_criteria, filter_embedding = criteria_name_and_embedding[criteria]
    # top n columns by embedding & data type
    results = cursor.execute(
        f"SELECT id FROM {embedding_db_name} WHERE has_eda_ops AND data_type=%s "
        f"ORDER BY {filter_criteria} <=> %s::vector LIMIT {n};",
        (data_type, str(filter_embedding)))

    top_columns_by_embedding = [result[0] for result in results.fetchall()]

    return_data = {'top_columns_by_embedding': top_columns_by_embedding}
    return flask.jsonify(return_data)


@app.route('/fetch_eda_operations', methods=['POST'])
def fetch_eda_operations():
    similar_column_id = request.get_json().get('similar_column_id', '')
    main_column_name = request.get_json().get('main_column_name', '')
    query_table_name = request.get_json().get('query_table_name')
    graphdb_repo = request.get_json().get('graphdb_repo', '')
    embedding_db_name = request.get_json().get('embedding_db_name', '')
    embedding_criteria = request.get_json().get('embedding_criteria', 'content_label')


    graphdb_endpoint = 'http://localhost:7200'
    graphdb_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}'

    univariate_eda_ops = []
    # Univariate EDA Ops
    graph_query = """
    PREFIX kglids: <http://kglids.org/ontology/>
    PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
    
    SELECT ?eda ?chart_type WHERE {
        ?eda a kglids:EDAOperation.
        ?eda pipeline:hasAnalysisType "univariate".
        ?eda pipeline:hasChartType ?chart_type.
        <http://kglids.org/resource/%s> pipeline:hasEDAOperation ?eda.
    }
    """ % similar_column_id

    results = query_graph(graph_query, graphdb_url)

    for result in results:
        univariate_eda_ops.append({'eda_id': result['eda']['value'],
                               'chart_type': result['chart_type']['value'],
                               'chart_columns': [main_column_name],
                               'grouping_column': None})

    # Bivariate  EDA Ops

    graph_query = """
    PREFIX kglids: <http://kglids.org/ontology/>
    PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
    PREFIX data: <http://kglids.org/ontology/data/>
    
    SELECT distinct ?eda ?chart_type ?similar_secondary_column ?similar_secondary_column_type WHERE {
        ?eda a kglids:EDAOperation.
        ?eda pipeline:hasAnalysisType "bivariate".
        ?eda pipeline:hasChartType ?chart_type.
        <http://kglids.org/resource/%s> pipeline:hasEDAOperation ?eda.
        ?similar_secondary_column pipeline:hasEDAOperation ?eda.
        ?similar_secondary_column a kglids:Column.
        ?similar_secondary_column data:hasDataType ?similar_secondary_column_type.
        FILTER (?similar_secondary_column != <http://kglids.org/resource/%s>).
    }
    """ % (similar_column_id, similar_column_id)
    results = query_graph(graph_query, graphdb_url)
    bivariate_eda_ops = []
    for result in results:
        bivariate_eda_ops.append({'eda_id': result['eda']['value'],
                               'chart_type': result['chart_type']['value'],
                               'chart_columns': [main_column_name],
                               'similar_secondary_column_id': result['similar_secondary_column']['value'],
                               'similar_secondary_column_type': result['similar_secondary_column_type']['value'],
                               'grouping_column': None})

    # find columns in query table that have the same data type as secondary_column and have closest embedding (must not be the main column)
    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    for eda_operation in bivariate_eda_ops:
        results = cursor.execute(
            f"""SELECT e.name FROM {embedding_db_name} e
                       WHERE e.dataset_name = 'query' 
                             AND e.table_name = '{query_table_name}'
                             AND e.data_type=%s
                             AND e.name != %s
                       ORDER BY e.{embedding_criteria}_embedding <=>  (
                            SELECT e2.{embedding_criteria}_embedding
                            FROM {embedding_db_name} e2
                            WHERE e2.id = %s)
                       LIMIT 1;""",
            (eda_operation['similar_secondary_column_type'], main_column_name,
             eda_operation['similar_secondary_column_id'].replace('http://kglids.org/resource/', '')))
        secondary_column_name = results.fetchone()
        if secondary_column_name:
            eda_operation['chart_columns'].append(secondary_column_name[0])
            eda_operation['chart_columns'] = sorted(eda_operation['chart_columns'])

    bivariate_eda_ops = [operation for operation in bivariate_eda_ops if len(operation['chart_columns']) > 1]

    # multivariate EDA ops

    graph_query = """
        PREFIX kglids: <http://kglids.org/ontology/>
        PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
        PREFIX data: <http://kglids.org/ontology/data/>
        
        SELECT ?eda ?chart_type (GROUP_CONCAT(?other_column_info; SEPARATOR = ",") AS ?other_columns)  
        WHERE {
            ?eda a kglids:EDAOperation.
            ?eda pipeline:hasAnalysisType "multivariate".
            ?eda pipeline:hasChartType ?chart_type.
            <http://kglids.org/resource/%s> pipeline:hasEDAOperation ?eda.
            ?other_column pipeline:hasEDAOperation ?eda.
            ?other_column data:hasDataType ?other_column_type .
            BIND(CONCAT(STR(?other_column), ";" , ?other_column_type) AS ?other_column_info).
            FILTER(?other_column != <http://kglids.org/resource/%s>) .
        }
        GROUP BY ?eda ?chart_type
    """ % (similar_column_id, similar_column_id)
    results = query_graph(graph_query, graphdb_url)
    multivariate_eda_ops = []
    for result in results:
        similar_columns_ids_and_types = [tuple(i.split(';')) for i in result['other_columns']['value'].split(',')]
        eda_operation = {'eda_id': result['eda']['value'],
                         'chart_type': result['chart_type']['value'],
                         'other_similar_columns': similar_columns_ids_and_types,
                         'chart_columns': [main_column_name],
                         'grouping_column': None}
        if result['chart_type']['value'] in ['heatmap', 'pairwise']:
            eda_operation['other_similar_columns'] = []
            eda_operation['chart_columns'] = []

        multivariate_eda_ops.append(eda_operation)

    # find columns in query table that have the same data type as secondary columns and have closest embedding (must not be the main column)
    # query vector DB
    for eda_operation in multivariate_eda_ops:
        for other_similar_column_id, other_similar_column_type in eda_operation['other_similar_columns']:
            chart_columns_str = [i.replace('%','%%').replace("'", "''") for i in eda_operation['chart_columns']]
            matched_columns_str = '(' + ','.join([f"'{i}'" for i in chart_columns_str]) + ')'
            results = cursor.execute(
                f"""SELECT e.name FROM {embedding_db_name} e
                       WHERE e.dataset_name = 'query' 
                            AND e.table_name = '{query_table_name}'
                            AND e.data_type=%s
                            AND e.name NOT IN {matched_columns_str}
                       ORDER BY e.{embedding_criteria}_embedding <=>  (
                            SELECT e2.{embedding_criteria}_embedding
                            FROM {embedding_db_name} e2
                            WHERE e2.id = %s)
                            LIMIT 1;""",
            (other_similar_column_type, other_similar_column_id.replace('http://kglids.org/resource/', '')))
            matched_column_name = results.fetchone()
            if matched_column_name:
                eda_operation['chart_columns'].append(matched_column_name[0])

        eda_operation['chart_columns'] = sorted(eda_operation['chart_columns'])
    # keep only successful multivariate EDA OPs. Either heatmap/pairwise or ones with more than two columns
    multivariate_eda_ops = [i for i in multivariate_eda_ops
                      if i['chart_type'] in ['heatmap', 'pairwise'] or len(i['chart_columns']) > 2]

    return flask.jsonify(univariate_eda_ops + bivariate_eda_ops + multivariate_eda_ops)


@app.route('/create_evaluation_graphs_and_databases', methods=['POST'])
def create_evaluation_graphs_and_databases():
    graphdb_endpoint = 'http://localhost:7200'

    graphdb_all_kaggle_repo = request.get_json().get('graphdb_repo', '')
    graphdb_autoeda_repo = request.get_json().get('graphdb_autoeda_repo', '')
    autoeda_dataset_ids = set(request.get_json().get('autoeda_dataset_ids', ''))
    test_dataset_ids = set(request.get_json().get('test_dataset_ids', ''))
    embedding_db_name = request.get_json().get('embedding_db_name', '')
    autoeda_embedding_db_name = request.get_json().get('autoeda_embedding_db_name', '')

    # A. GraphDB Repos
    print('Creating GraphDB repos...')
    create_graphdb_repo(graphdb_endpoint, graphdb_autoeda_repo)

    data_source_query = """
    PREFIX kglids: <http://kglids.org/ontology/>
    select ?source 
    where{
        ?source a kglids:Source
    }
    """
    result = query_graph(data_source_query, f'{graphdb_endpoint}/repositories/{graphdb_all_kaggle_repo}')

    data_source_uri = result[0]['source']['value']
    # 1. copy dataset subgraph from default graphs
    print("Populating dataset subgraphs for", graphdb_autoeda_repo)
    args = [(data_source_uri, dataset_id, graphdb_endpoint, graphdb_all_kaggle_repo, graphdb_autoeda_repo)
            for dataset_id in autoeda_dataset_ids]
    pool = mp.Pool()
    list(tqdm(pool.imap_unordered(upload_dataset_subgraph_to_evaluation_graph, args), total=len(args)))


    # 2. copy pipeline graphs
    query = """
      SELECT distinct ?graph
      WHERE {
          GRAPH ?graph {
              ?s ?p ?o.
          }
      }
      """
    graph_list = query_graph(
        query, f"{graphdb_endpoint}/repositories/{graphdb_all_kaggle_repo}"
    )
    autoeda_graphs = []
    for graph in graph_list:
        graph_id = graph["graph"]["value"]
        graph_dataset = graph_id.replace(f"{data_source_uri}/", "")
        graph_dataset = graph_dataset.split("/")[0]
        if graph_dataset in autoeda_dataset_ids:
            autoeda_graphs.append(graph_id)


    print("populating pipeline graphs for", graphdb_autoeda_repo)
    args = [(graph, graphdb_endpoint, graphdb_all_kaggle_repo, graphdb_autoeda_repo) for graph in autoeda_graphs]
    pool = mp.Pool()
    list(tqdm(pool.imap_unordered(upload_pipeline_subgraph_to_evaluation_graph, args), total=len(args)))

    # B. Postgres DBs
    print('Creating Embedding database')
    create_evaluation_embedding_dbs(test_dataset_ids, embedding_db_name, autoeda_embedding_db_name)
    return 'Success!'


initialize_autoeda()

if __name__ == '__main__':
    pass
    app.run(host='127.0.0.1', port=8080)

