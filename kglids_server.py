import sys
sys.path.append('./kg_governor/data_profiling/src')
import os
import math
import fasttext
import flask
from flask.globals import request
import pandas as pd
import psycopg
import SPARQLWrapper

from server_utils import query_graph
from kg_governor.data_profiling.src.fine_grained_type_detector import FineGrainedColumnTypeDetector
from kg_governor.data_profiling.src.profile_creators.profile_creator import ProfileCreator
from kg_governor.data_profiling.src.model.table import Table


flask_app = flask.Flask(__name__)

graph_endpoint = 'http://localhost:7200/repositories/kaggle_eda'

@flask_app.route('/profile_column', methods=['POST'])
def profile_column():
    pass


@flask_app.route('/find_similar_columns', methods=['POST'])
def find_similar_columns():
    fasttext_path = os.path.expanduser('./kg_governor/data_profiling/src/fasttext_embeddings/cc.en.300.bin')
    embedding_db_name = 'kaggle_eda_column_embeddings'

    column_name = request.get_json().get('column_name', '')
    column_values = request.get_json().get('column_values', '')

    # profile the column
    column = pd.Series(column_values)
    column = pd.to_numeric(column, errors='ignore')
    column = column.convert_dtypes()
    column = column.astype(str) if column.dtype == object else column

    column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column)
    column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, Table('_', '_', '_'))
    column_profile = column_profile_creator.create_profile()

    if column_profile.get_embedding():
        content_embedding = column_profile.get_embedding() #+ [math.sqrt(column_profile.get_embedding_scaling_factor())] + [0]
    else:
        # boolean columns
        content_embedding = [0]*300 #1 + [column_profile.get_true_ratio()]

    ft = fasttext.load_model(fasttext_path)
    sanitized_name = column_name.replace('\n', ' ').replace('_', ' ').strip()
    label_embedding = ft.get_sentence_vector(sanitized_name).tolist()

    # query vector DB
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres')
    cursor = conn.cursor()

    print(column_type)
    # top 3 columns by embedding & data type
    results = cursor.execute(f'SELECT id FROM {embedding_db_name} WHERE data_type=%s ORDER BY content_embedding <=> %s::vector LIMIT 3;',
                             (column_type.value, str(content_embedding)))

    top3_columns_by_content_embedding = [result[0] for result in results.fetchall()]
    # top 3 columns by name & data type
    results = cursor.execute(f'SELECT id FROM {embedding_db_name} WHERE data_type=%s ORDER BY label_embedding <=> %s::vector LIMIT 3;',
                             (column_type.value, str(label_embedding)))
    top3_columns_by_label_embedding = [result[0] for result in results.fetchall()]
    # top 3 columns by embedding + name & data type
    results = cursor.execute(f'SELECT id FROM {embedding_db_name} WHERE data_type=%s ORDER BY content_label_embedding <=> %s::vector LIMIT 3;',
                             (column_type.value, str(content_embedding + label_embedding)))
    top3_columns_by_content_label_embedding = [result[0] for result in results.fetchall()]

    return_data = {'top3_by_content': top3_columns_by_content_embedding,
                   'top3_by_label': top3_columns_by_label_embedding,
                   'top3_by_content_label': top3_columns_by_content_label_embedding}
    return flask.jsonify(return_data)


@flask_app.route('/fetch_eda_operations', methods=['POST'])
def fetch_eda_operations():

    main_column_id = request.get_json().get('main_column_name', '')
    secondary_column_id = request.get_json().get('secondary_column_name', '')
    analysis_type = request.get_json().get('analysis_type', '')

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
        """ % main_column_id


    elif analysis_type == 'bivariate':
        pass


if __name__ == '__main__':
    flask_app.run(host='127.0.0.1', port=8080)
