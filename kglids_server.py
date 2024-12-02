import sys

sys.path.append('./kg_governor/data_profiling/src')
import fasttext
import flask
from flask.globals import request
import pandas as pd
import psycopg
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp

from server_utils import query_graph, upload_graph, get_graph_content, create_evaluation_embedding_dbs
from storage.utils.populate_graphdb import create_or_replace_repo
from kg_governor.data_profiling.src.fine_grained_type_detector import FineGrainedColumnTypeDetector
from kg_governor.data_profiling.src.profile_creators.profile_creator import ProfileCreator
from kg_governor.data_profiling.src.model.column_profile import ColumnProfile
from kg_governor.data_profiling.src.model.table import Table

flask_app = flask.Flask(__name__)

fasttext_path = './storage/embeddings/cc.en.300.bin'
ft = fasttext.load_model(fasttext_path)


def profile_column(args):
    # profiles each column by analyzing its fine-grained type and generating the embedding
    df, column_name = args
    # profile the column and generate its embeddings
    column = pd.to_numeric(df[column_name], errors='ignore')
    column = column.convert_dtypes()
    column = column.astype(str) if column.dtype == object else column

    column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column)
    column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, Table('query',
                                                                                           'query.csv',
                                                                                           'query'))
    column_profile: ColumnProfile = column_profile_creator.create_profile()

    if column_profile.get_embedding():
        content_embedding = column_profile.get_embedding()
    else:
        # boolean columns
        content_embedding = [column_profile.get_true_ratio()] * 300

    sanitized_name = column_name.replace('\n', ' ').replace('_', ' ').strip()
    label_embedding = ft.get_sentence_vector(sanitized_name).tolist()

    content_label_embedding = content_embedding + label_embedding
    column_info = {'column_id': column_profile.get_column_id(),
                   'column_name': column_profile.get_column_name(),
                   'data_type': column_profile.get_data_type(),
                   'unique_values_count': column_profile.get_distinct_values_count(),
                   'dataset_name': column_profile.get_dataset_name(),
                   'table_name': column_profile.get_table_name(),
                   'content_embedding': str(content_embedding),
                   'label_embedding': str(label_embedding),
                   'content_label_embedding': str(content_label_embedding)}
    return column_info


@flask_app.route('/profile_query_table', methods=['POST'])
def profile_query_table():
    embedding_db_name = request.args.get('embedding_db_name')
    print(embedding_db_name)
    file = request.files['query_table']
    df = pd.read_csv(file)

    print(datetime.now(), 'Received CSV file. Profiling and storing embeddings...')
    profile_args = [(df, column_name) for column_name in df.columns]
    pool = mp.Pool()
    column_info = list(tqdm(pool.imap_unordered(profile_column, profile_args), total=len(profile_args)))

    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    # remove previous query table if exists
    cursor.execute(f"DELETE FROM {embedding_db_name} WHERE dataset_name = 'query' AND table_name = 'query.csv'")
    # add new query table to embedding store
    insert_query = f'''INSERT INTO {embedding_db_name} (id, name, data_type, dataset_name, table_name, 
                            content_embedding, label_embedding, content_label_embedding) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);'''
    insert_data = [(i['column_id'], i['column_name'], i['data_type'], i['dataset_name'], i['table_name'],
                    i['content_embedding'], i['label_embedding'], i['content_label_embedding'])
                   for i in column_info]
    cursor.executemany(insert_query, insert_data)
    cursor.close()

    return_data = [{k: i[k] for k in ['column_name', 'data_type', 'unique_values_count']} for i in column_info]
    return flask.jsonify(return_data)


@flask_app.route('/find_similar_columns', methods=['POST'])
def find_similar_columns():

    column_name = request.get_json().get('main_column_name', '')
    embedding_db_name = request.get_json().get('embedding_db_name', '')

    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres')
    cursor = conn.cursor()
    column_fetch_query = f"""SELECT data_type, content_embedding, label_embedding, content_label_embedding
                            FROM {embedding_db_name} 
                            WHERE dataset_name = 'query' AND table_name = 'query.csv' and name = '{column_name.replace("'", "''")}'"""
    results = cursor.execute(column_fetch_query)
    data_type, content_embedding, label_embedding, content_label_embedding = results.fetchone()
    # top 3 columns by embedding & data type
    results = cursor.execute(
        f"SELECT id FROM {embedding_db_name} WHERE dataset_name != 'query' AND table_name != 'query.csv' AND data_type=%s ORDER BY content_embedding <=> %s::vector LIMIT 3;",
        (data_type, str(content_embedding)))

    top3_columns_by_content_embedding = [result[0] for result in results.fetchall()]
    # top 3 columns by name & data type
    results = cursor.execute(
        f"SELECT id FROM {embedding_db_name} WHERE dataset_name != 'query' AND table_name != 'query.csv' AND data_type=%s ORDER BY label_embedding <=> %s::vector LIMIT 3;",
        (data_type, str(label_embedding)))
    top3_columns_by_label_embedding = [result[0] for result in results.fetchall()]
    # top 3 columns by embedding + name & data type
    results = cursor.execute(
        f"SELECT id FROM {embedding_db_name} WHERE dataset_name != 'query' AND table_name != 'query.csv' AND data_type=%s ORDER BY content_label_embedding <=> %s::vector LIMIT 3;",
        (data_type, str(content_label_embedding)))
    top3_columns_by_content_label_embedding = [result[0] for result in results.fetchall()]

    return_data = {'top3_by_content': top3_columns_by_content_embedding,
                   'top3_by_label': top3_columns_by_label_embedding,
                   'top3_by_content_label': top3_columns_by_content_label_embedding}
    return flask.jsonify(return_data)


@flask_app.route('/fetch_eda_operations', methods=['POST'])
def fetch_eda_operations():
    similar_column_id = request.get_json().get('similar_column_id', '')
    main_column_name = request.get_json().get('main_column_name', '')
    analysis_type = request.get_json().get('analysis_type', '')
    graphdb_repo = request.get_json().get('graphdb_repo', '')
    embedding_db_name = request.get_json().get('embedding_db_name', '')

    graphdb_endpoint = 'http://localhost:7200'
    graphdb_url = f'{graphdb_endpoint}/repositories/{graphdb_repo}'

    if analysis_type == 'univariate':
        graph_query = """
        PREFIX kglids: <http://kglids.org/ontology/>
        PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
        
        SELECT ?eda ?chart_type WHERE {
            ?eda a kglids:EDAOperation.
            ?eda pipeline:hasAnalysisType "univariate".
            ?eda pipeline:hasChartType ?chart_type.
            ?eda pipeline:hasMainColumn <http://kglids.org/resource/%s>.
        }
        """ % similar_column_id

        results = query_graph(graph_query, graphdb_url)
        eda_operations = []
        for result in results:
            eda_operations.append({'eda_id': result['eda']['value'], 'chart_type': result['chart_type']['value']})

    elif analysis_type == 'bivariate':
        graph_query = """
        PREFIX kglids: <http://kglids.org/ontology/>
        PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
        PREFIX data: <http://kglids.org/ontology/data/>
        
        SELECT distinct ?eda ?chart_type ?similar_secondary_column ?similar_secondary_column_type WHERE {
            ?eda a kglids:EDAOperation.
            ?eda pipeline:hasAnalysisType "bivariate".
            ?eda pipeline:hasChartType ?chart_type.
            ?eda ?p1 <http://kglids.org/resource/%s>.
            ?eda ?p2 ?similar_secondary_column .
            ?similar_secondary_column a kglids:Column.
            ?similar_secondary_column data:hasDataType ?similar_secondary_column_type.
            FILTER (?p1 != ?p2).
            FILTER (?similar_secondary_column != <http://kglids.org/resource/%s>).
        }
        """ % (similar_column_id, similar_column_id)
        results = query_graph(graph_query, graphdb_url)
        eda_operations = []
        for result in results:
            eda_operations.append({'eda_id': result['eda']['value'],
                                   'chart_type': result['chart_type']['value'],
                                   'similar_secondary_column_id': result['similar_secondary_column']['value'],
                                   'similar_secondary_column_type': result['similar_secondary_column_type']['value']})

        # find columns in query table that have the same data type as secondary_column and have closest embedding (must not be the main column)
        # query vector DB
        conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
        cursor = conn.cursor()
        for eda_operation in eda_operations:
            results = cursor.execute(
                f"""SELECT e.name FROM {embedding_db_name} e
                           WHERE e.dataset_name = 'query' 
                                 AND e.table_name = 'query.csv'
                                 AND e.data_type=%s
                                 AND e.name != %s
                           ORDER BY e.content_label_embedding <=>  (
                                SELECT e2.content_label_embedding
                                FROM {embedding_db_name} e2
                                WHERE e2.id = %s)
                           LIMIT 1;""",
                (eda_operation['similar_secondary_column_type'], main_column_name,
                 eda_operation['similar_secondary_column_id'].replace('http://kglids.org/resource/', '')))
            secondary_column_name = results.fetchone()
            if secondary_column_name:
                eda_operation['secondary_column_name'] = secondary_column_name[0]

        eda_operations = [operation for operation in eda_operations if 'secondary_column_name' in operation]

    elif analysis_type == 'multivariate':

        graph_query = """
            PREFIX kglids: <http://kglids.org/ontology/>
            PREFIX pipeline: <http://kglids.org/ontology/pipeline/>
            PREFIX data: <http://kglids.org/ontology/data/>
            
            SELECT distinct ?eda ?chart_type ?similar_main_column ?similar_main_column_type ?similar_secondary_column ?similar_secondary_column_type ?similar_grouping_column ?similar_grouping_column_type  WHERE {
                ?eda a kglids:EDAOperation.
                ?eda pipeline:hasAnalysisType "multivariate".
                ?eda pipeline:hasChartType ?chart_type.
                ?eda ?p1 <http://kglids.org/resource/%s>.
                ?eda pipeline:hasMainColumn ?similar_main_column .
                ?eda pipeline:hasSecondaryColumn ?similar_secondary_column .
                ?eda pipeline:hasGroupingColumn ?similar_grouping_column .
                ?similar_main_column data:hasDataType ?similar_main_column_type .
                ?similar_secondary_column data:hasDataType ?similar_secondary_column_type.
                ?similar_grouping_column data:hasDataType ?similar_grouping_column_type.
                FILTER(?similar_main_column != ?similar_secondary_column) .
                FILTER(?similar_secondary_column != ?similar_grouping_column) .
                FILTER(?similar_main_column != ?similar_grouping_column) .
            }
        """ % (similar_column_id)
        results = query_graph(graph_query, graphdb_url)
        eda_operations = []
        for result in results:
            # check which one is has the similar_column_id
            # have it
            similar_column_ids_types = sorted([(result['similar_secondary_column']['value'],
                                                result['similar_secondary_column_type']['value']),
                                               (result['similar_grouping_column']['value'],
                                                result['similar_grouping_column_type']['value'])
                                               ], key=lambda x: x[0])
            eda_operations.append({'eda_id': result['eda']['value'],
                                   'chart_type': result['chart_type']['value'],
                                   'similar_secondary_column_id': similar_column_ids_types[0][0],
                                   'similar_secondary_column_type': similar_column_ids_types[0][1],
                                   'similar_grouping_column_id': similar_column_ids_types[1][0],
                                   'similar_grouping_column_type': similar_column_ids_types[1][1]})

        # find columns in query table that have the same data type as secondary columns and have closest embedding (must not be the main column)
        # query vector DB
        conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
        cursor = conn.cursor()
        for eda_operation in eda_operations:
            results = cursor.execute(
                f"""SELECT e.name FROM {embedding_db_name} e
                               WHERE e.dataset_name = 'query' 
                                     AND e.table_name = 'query.csv'
                                     AND e.data_type=%s
                                     AND e.name != %s
                               ORDER BY e.content_label_embedding <=>  (
                                    SELECT e2.content_label_embedding
                                    FROM {embedding_db_name} e2
                                    WHERE e2.id = %s)
                               LIMIT 1;""",
                (eda_operation['similar_secondary_column_type'], main_column_name,
                 eda_operation['similar_secondary_column_id'].replace('http://kglids.org/resource/', '')))
            secondary_column_name = results.fetchone()

            if not secondary_column_name:
                continue
            eda_operation['secondary_column_name'] = secondary_column_name[0]

            results = cursor.execute(
                f"""SELECT e.name FROM {embedding_db_name} e
                                           WHERE e.dataset_name = 'query' 
                                                 AND e.table_name = 'query.csv'
                                                 AND e.data_type=%s
                                                 AND e.name != %s
                                                 AND e.name != %s
                                           ORDER BY e.content_label_embedding <=>  (
                                                SELECT e2.content_label_embedding
                                                FROM {embedding_db_name} e2
                                                WHERE e2.id = %s)
                                           LIMIT 1;""",
                (
                eda_operation['similar_grouping_column_type'], main_column_name, eda_operation['secondary_column_name'],
                eda_operation['similar_grouping_column_id'].replace('http://kglids.org/resource/', '')))
            grouping_column_name = results.fetchone()
            if not grouping_column_name:
                continue
            eda_operation['grouping_column_name'] = grouping_column_name[0]

        eda_operations = [operation for operation in eda_operations
                          if 'secondary_column_name' in operation and 'grouping_column_name' in operation]

    else:
        eda_operations = []
    return flask.jsonify(eda_operations)


@flask_app.route('/create_evaluation_graphs_and_databases', methods=['POST'])
def create_evaluation_graphs_and_databases():
    graphdb_endpoint = 'http://localhost:7200'
    graphdb_all_kaggle_repo = 'kaggle_eda'

    graphdb_autoeda_repo = request.get_json().get('graphdb_autoeda_repo', '')
    autoeda_dataset_ids = set(request.get_json().get('autoeda_dataset_ids', ''))
    test_dataset_ids = set(request.get_json().get('test_dataset_ids', ''))
    embedding_db_name = request.get_json().get('embedding_db_name', '')
    autoeda_embedding_db_name = request.get_json().get('autoeda_embedding_db_name', '')

    # A. GraphDB Repos
    print('Creating GraphDB repos...')
    create_or_replace_repo(graphdb_endpoint, graphdb_autoeda_repo)

    # 1. copy dataset subgraph from default graphs
    print("Populating dataset subgraphs for", graphdb_autoeda_repo)
    for dataset_id in tqdm(autoeda_dataset_ids):
        dataset_uri = f"<http://kglids.org/resource/kaggle/{dataset_id}>"
        graph_query = """
        PREFIX kglids: <http://kglids.org/ontology/>
        construct {?s ?p ?o}
        WHERE {
        ?s kglids:isPartOf+ %s.
        ?s ?p ?o . 
        } """ % dataset_uri
        graph_content = get_graph_content(
            graphdb_endpoint, graphdb_all_kaggle_repo, graph_query=graph_query
        )
        upload_graph(graph_content, graphdb_endpoint, graphdb_autoeda_repo)

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
        graph_dataset = graph_id.replace("http://kglids.org/resource/kaggle/", "")
        graph_dataset = graph_dataset.split("/")[0]
        if graph_dataset in autoeda_dataset_ids:
            autoeda_graphs.append(graph_id)


    print("populating pipeline graphs for", graphdb_autoeda_repo)
    for graph in tqdm(autoeda_graphs):
        graph_content = get_graph_content(
            graphdb_endpoint, graphdb_all_kaggle_repo, named_graph_uri=graph
        )
        upload_graph(graph_content, graphdb_endpoint, graphdb_autoeda_repo, graph)

    # B. Postgres DBs
    print('Creating Embedding database')
    create_evaluation_embedding_dbs(test_dataset_ids, embedding_db_name, autoeda_embedding_db_name)
    return 'Success!'


if __name__ == '__main__':
    flask_app.run(host='127.0.0.1', port=8080)
