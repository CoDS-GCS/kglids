import ast
import json
import os
import pandas as pd
import requests
import time
import re
import util
from typing import Dict

from src.datatypes import Library
from src.adapters.repository import Neo4JRepository
from src.pipeline_abstraction import NodeVisitor
from src.datatypes import GraphInformation

libraries: Dict[str, Library]
libraries = dict()
default_graph = []


def main():
    overall_start = time.time()

    dataset: os.DirEntry
    # loop through datasets & pipelines
    for dataset in os.scandir('../data/kaggle'):
        pipeline: os.DirEntry
        if dataset.is_dir():
            working_file = {}
            table: os.DirEntry
            if not os.path.isdir(f'{dataset.path}/data/'):
                continue
            for table in os.scandir(f'{dataset.path}/data/'):
                if table.name != '.DS_Store':
                    try:
                        working_file[table.name] = pd.read_csv(table.path, nrows=1)
                    except Exception as e:
                        print("-<>", table.name, e)
            if not os.path.isdir(f'{dataset.path}/notebooks/'):
                continue
            for pipeline in os.scandir(f'{dataset.path}/notebooks'):
                if pipeline.is_dir():
                    try:
                        with open(f'{pipeline.path}/pipeline_info.json', 'r') as f:
                            pipeline_info = json.load(f)
                        with open(f'{pipeline.path}/kernel-metadata.json', 'r') as f:
                            metadata = json.load(f)
                        pipeline_info['tags'] = metadata['keywords']
                        file: os.DirEntry
                        for file in os.scandir(pipeline.path):
                            if '.py' in file.name:
                                pipeline_analysis(working_file,
                                                  file.name.split('.')[0],
                                                  dataset,
                                                  file.path,
                                                  pipeline.name,
                                                  pipeline_info,
                                                  metadata['id'].replace('/', '.'))
                    except FileNotFoundError as e:
                        continue

    # Default graph creation
    neo4j_repository = Neo4JRepository(
        source="",
        dataset_name="",
        python_file_name=""
    )
    neo4j_repository.delete_everything()
    neo4j_repository.database_init()
    for el in default_graph:
        neo4j_repository.create_pipeline_node(el)

    libs = [library.str() for library in libraries.values()]
    neo4j_repository.create_library_nodes(libs)

    rdf_host = "http://localhost:7474/rdf"
    post_pstfx = "/neo4j/cypher"
    credentials = ("neo4j", "123456")

    post_url = rdf_host + post_pstfx
    post_data = {
        "cypher": "MATCH (n) OPTIONAL MATCH (n)-[r]->(p) RETURN n,r,p",
        "cypherParams": {},
        "format": "Turtle*"
    }
    post_response = requests.post(post_url, auth=credentials, json=post_data)

    literals = {}
    rdf_file = ''
    sentence = ''
    previous = ''
    new_line_count = 0
    for char in post_response.content:
        value = chr(char)
        sentence = sentence + value
        if (value == '.' and previous == ' ') or (value == '\n' and previous in ('.', "\n")):
            if sentence.startswith("<neo4j://graph"):
                match = re.match(r'(<neo4j://graph.individuals#[0-9]+>).*\n.*n4sch:value (".*")', sentence)
                if match:
                    literals[match.group(1)] = match.group(2)
            elif sentence.startswith("@prefix n4sch:"):
                pass
            elif sentence == '\n':
                if new_line_count < 2:
                    rdf_file += sentence
                    new_line_count += 1
            else:
                rdf_file = rdf_file + sentence
                new_line_count = 0
            sentence = ""
        previous = value

    for node, value in literals.items():
        rdf_file = rdf_file.replace(node, value)

    with open(f'../kaggle_rdf/default.ttl', 'w') as output_file:
        output_file.write(rdf_file)

    overall_end = time.time()
    print(overall_end - overall_start)


def pipeline_analysis(working_file, f, dataset: os.DirEntry, file_path, pipeline_name, pipeline_info, output_filename):
    starting_time = time.time()
    SOURCE = 'kaggle'
    DATASET_NAME = dataset.name
    PYTHON_FILE_NAME = output_filename

    # Read pipeline file
    with open(file_path, 'r') as src_file:
        src = src_file.read()

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        with open(f'./errors.csv', 'a') as output_file:
            output_file.write(f'{PYTHON_FILE_NAME},{e.msg}\n')
        return

    # Initialize graph information linked list and Node Visitor
    graph = GraphInformation(PYTHON_FILE_NAME, SOURCE, DATASET_NAME, libraries)
    node_visitor = NodeVisitor(graph_information=graph)
    node_visitor.working_file = working_file

    # Pipeline analysis
    node_visitor.visit(tree)
    ending_time = time.time()
    print("Processing time: ", ending_time - starting_time, 'seconds')

    # Datastructures preparation for insertion to Neo4j
    file_elements = [el.str() for el in graph.files.values()]
    nodes = []
    head = graph.head
    line = 1
    while head is not None:
        head.generate_uri(SOURCE, DATASET_NAME, PYTHON_FILE_NAME, line)
        line += 1
        head = head.next
    head = graph.head
    while head is not None:
        nodes.append(head.str())
        head = head.next

    neo4j_start = time.time()
    neo4j_repository = Neo4JRepository(
        source=SOURCE,
        dataset_name=DATASET_NAME,
        python_file_name=PYTHON_FILE_NAME
    )
    # Database preparation
    neo4j_repository.delete_everything()
    neo4j_repository.database_init()
    pipeline_info['uri'] = util.create_pipeline_uri(SOURCE, DATASET_NAME, output_filename)
    pipeline_info['dataset'] = util.create_dataset_uri(SOURCE, DATASET_NAME)
    default_graph.append(pipeline_info)

    # Pipelines insertion to Neo4j
    neo4j_repository.create_file_nodes(file_elements)

    if len(nodes) > 2:
        neo4j_repository.create_first_node(nodes[0])
        neo4j_repository.create_node_nodes(nodes[1:])
    neo4j_end = time.time()
    print("Neo4j time:", neo4j_end - neo4j_start, 'seconds')

    n10s_start = time.time()
    # Neo4j to RDF
    rdf_host = "http://localhost:7474/rdf"
    post_pstfx = "/neo4j/cypher"
    credentials = ("neo4j", "123456")

    post_url = rdf_host + post_pstfx
    post_data = {
        "cypher": "MATCH p=()-[]->() RETURN p",
        "cypherParams": {},
        "format": "Turtle*"
    }
    post_response = requests.post(post_url, auth=credentials, json=post_data)

    literals = {}
    rdf_file = ''
    sentence = ''
    previous = ''
    new_line_count = 0

    # RDF hacking
    for char in post_response.content:
        value = chr(char)
        sentence = sentence + value
        if (value == '.' and previous == ' ') or (value == '\n' and previous in ('.', "\n")):
            if sentence.startswith("<neo4j://graph"):
                match = re.match(r'(<neo4j://graph.individuals#[0-9]+>).*\n.*n4sch:value (".*")', sentence)
                if match:
                    literals[match.group(1)] = match.group(2)
            elif sentence.startswith("@prefix n4sch:"):
                pass
            elif sentence == '\n':
                if new_line_count < 2:
                    rdf_file += sentence
                    new_line_count += 1
            else:
                rdf_file = rdf_file + sentence
                new_line_count = 0
            sentence = ""
        previous = value

    for node, value in literals.items():
        rdf_file = rdf_file.replace(node, value)

    # File creation and pipeline output
    if not os.path.isdir(f'../kaggle_rdf/{dataset.name}'):
        os.mkdir(f'../kaggle_rdf/{dataset.name}')

    with open(f'../kaggle_rdf/{dataset.name}/{output_filename}.ttl', 'w') as output_file:
        output_file.write(rdf_file)

    n10s_end = time.time()
    with open(f'../stats/statistics_{dataset.name}.csv', 'a') as output_file:
        output_file.write(f'{PYTHON_FILE_NAME},{n10s_end - starting_time}\n')
    print("Neosemantic processing:", n10s_end - n10s_start, "seconds")
    print("Total time:", n10s_end - starting_time, "seconds")


if __name__ == "__main__":
    main()
