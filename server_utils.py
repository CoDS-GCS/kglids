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


def create_evaluation_embedding_dbs(test_dataset_ids, embedding_db_name, autoeda_embedding_db_name):
    copy_embedding_db(embedding_db_name, autoeda_embedding_db_name)

    test_dataset_ids_literal = '(' + ','.join([f"'{i}'" for i in test_dataset_ids]) + ')'
    conn = psycopg.connect(dbname=autoeda_embedding_db_name, user='postgres', password='postgres', autocommit=True)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {autoeda_embedding_db_name} WHERE dataset_name IN {test_dataset_ids_literal} ;")
    conn.close()
    print('Evaluation datasets created')