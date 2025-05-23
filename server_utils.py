from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import psycopg


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


def get_graph_content(
        graphdb_endpoint, graphdb_repo, graph_query=None, named_graph_uri=None
):
    headers = {
        "Accept": "application/x-binary-rdf",
    }
    if named_graph_uri:
        download_url = f"{graphdb_endpoint}/repositories/{graphdb_repo}/rdf-graphs/service?graph={named_graph_uri}"
        response = requests.get(download_url, headers=headers)
    else:
        headers["Conetnt-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        data = {"query": graph_query}
        download_url = f"{graphdb_endpoint}/repositories/{graphdb_repo}"
        response = requests.get(download_url, headers=headers, params=data)

    if response.status_code // 100 != 2:
        print("Error downloading graph:", named_graph_uri, ":", response.text)
        print('QUERY:', graph_query)
        print('NAMED GRAPH URI:', named_graph_uri)
        return None
    return response.content


def upload_graph(
        file_content, graphdb_endpoint, graphdb_repo, named_graph_uri=None
):
    headers = {
        "Content-Type": "application/x-binary-rdf",
        "Accept": "application/json",
    }

    upload_url = f"{graphdb_endpoint}/repositories/{graphdb_repo}/statements"
    if named_graph_uri:
        upload_url = f"{graphdb_endpoint}/repositories/{graphdb_repo}/rdf-graphs/service?graph={named_graph_uri}"

    response = requests.post(upload_url, headers=headers, data=file_content)
    if response.status_code // 100 != 2:
        print("Error uploading file:", ". Error:", response.text)


def add_has_eda_ops_column_to_embedding_db(embedding_db_name, graphdb_endpoint):
    # 1. check if embedding db has the column
    db_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='{embedding_db_name}' and column_name='has_eda_ops';
    """
    conn = psycopg.connect(dbname=embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    results = cursor.execute(db_query).fetchone()
    if results:
        # check if it has any true values
        db_query = f"""
        SELECT * from {embedding_db_name} WHERE has_eda_ops LIMIT 1;
        """
        results = cursor.execute(db_query).fetchone()
        if results:
            # database already has the column and is populated
            return

    print('Creating and populating has_eda_ops column in', embedding_db_name)
    # get list of columns with EDA operations
    graph_query = """
        PREFIX kglids: <http://kglids.org/ontology/>
        PREFIX pipeline: <http://kglids.org/ontology/pipeline/>

        SELECT distinct ?col
        WHERE {
            ?col a kglids:Column.
            ?col pipeline:hasEDAOperation ?eda.   
        }
    """
    results = query_graph(graph_query, graphdb_endpoint)
    column_uris = [result['col']['value'] for result in results]
    column_ids = [column_uri.split('/resource/')[1] for column_uri in column_uris]

    if not column_uris:
        print('No EDA operations in the graph. Terminating ...')
        exit()
    # add has_eda_ops column and populate it
    db_query = f"""
        ALTER TABLE {embedding_db_name}
        ADD COLUMN has_eda_ops BOOLEAN DEFAULT FALSE;
    """
    cursor.execute(db_query)
    column_ids_literal = '(' + ','.join([f"'{i}'" for i in column_ids]) + ')'
    db_query = f"""
        UPDATE {embedding_db_name}
        SET has_eda_ops = true
        WHERE id IN {column_ids_literal};
    """
    cursor.execute(db_query)
    print('Column has_eda_ops populated in', embedding_db_name)

def copy_embedding_db(source_db_name, target_db_name):
    conn = psycopg.connect(dbname='postgres', user='postgres', password='postgres', autocommit=True)

    cursor = conn.cursor()
    query = f"""
    SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity 
    WHERE pg_stat_activity.datname = '{source_db_name}' AND pid <> pg_backend_pid();
    """
    cursor.execute(query)
    query = f"""
        SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity 
        WHERE pg_stat_activity.datname = '{target_db_name}' AND pid <> pg_backend_pid();
        """
    cursor.execute(query)

    cursor.execute(f'DROP DATABASE IF EXISTS {target_db_name};')
    cursor.execute(f"CREATE DATABASE {target_db_name} WITH TEMPLATE {source_db_name} OWNER postgres;")

    conn = psycopg.connect(dbname=target_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    cursor.execute(f'ALTER TABLE {source_db_name} RENAME TO {target_db_name}')


def upload_dataset_subgraph_to_evaluation_graph(args):
    data_source_uri, dataset_id, graphdb_endpoint, graphdb_all_kaggle_repo, graphdb_autoeda_repo = args
    dataset_uri = f"<{data_source_uri}/{dataset_id}>"
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


def upload_pipeline_subgraph_to_evaluation_graph(args):
    graph, graphdb_endpoint, graphdb_all_kaggle_repo, graphdb_autoeda_repo = args
    graph_content = get_graph_content(
        graphdb_endpoint, graphdb_all_kaggle_repo, named_graph_uri=graph
    )
    upload_graph(graph_content, graphdb_endpoint, graphdb_autoeda_repo, graph)


def create_evaluation_embedding_dbs(test_dataset_ids, embedding_db_name, autoeda_embedding_db_name):
    copy_embedding_db(embedding_db_name, autoeda_embedding_db_name)

    test_dataset_ids_literal = '(' + ','.join([f"'{i}'" for i in test_dataset_ids]) + ')'
    conn = psycopg.connect(dbname=autoeda_embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {autoeda_embedding_db_name} WHERE dataset_name IN {test_dataset_ids_literal} ;")
    conn.close()
    print('Evaluation datasets created:', autoeda_embedding_db_name)