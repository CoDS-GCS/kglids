import ast
import json
import os
import pandas as pd
import time

import src.Calls as Calls
import src.util as util
from typing import Dict

from src.datatypes import Library
from src.pipeline_abstraction import NodeVisitor
from src.datatypes import GraphInformation
from src.json_to_rdf import build_pipeline_rdf_page, build_default_rdf_page, build_library_rdf_page

libraries: Dict[str, Library]
libraries = dict()
default_graph = []


def main():
    overall_start = time.time()
    
    OUTPUT_PATH = '../../../storage/knowledge_graph/pipelines_and_libraries/'

    dataset: os.DirEntry
    # loop through datasets & pipelines
    for dataset in os.scandir('../data/kaggle'):
        pipeline: os.DirEntry
        if dataset.is_dir():
            working_file = {}
            table: os.DirEntry
            if not os.path.isdir(f'../kaggle_small/{dataset.name}'):
                continue
            for table in os.scandir(f'../kaggle_small/{dataset.name}'):
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
                                pipeline_analysis(working_file=working_file,
                                                  dataset=dataset,
                                                  file_path=file.path,
                                                  pipeline_info=pipeline_info,
                                                  output_path=OUTPUT_PATH,
                                                  output_filename=metadata['id'].replace('/', '.'))
                    except FileNotFoundError as e:
                        continue

    libs = [library.str() for library in libraries.values()]
    # with open('kaggle/library.json', 'w') as f:
    #     json.dump(libs, f)
    #
    # with open('kaggle/default.json', 'w') as f:
    #     json.dump(default_graph, f)

    with open(os.path.join(OUTPUT_PATH, 'library.ttl'), 'w') as f:
        f.write(build_library_rdf_page(libs))

    with open(os.path.join(OUTPUT_PATH, 'default.ttl'), 'w') as f:
        f.write(build_default_rdf_page(default_graph))


    # literals = {}
    # rdf_file = ''
    #
    # for node, value in literals.items():
    #     rdf_file = rdf_file.replace(node, value)
    #
    # with open(f'../kaggle_rdf/default.ttl', 'w') as output_file:
    #     output_file.write(rdf_file)

    overall_end = time.time()
    print(overall_end - overall_start)
    print(Calls.read_csv_call.count)


def pipeline_analysis(working_file, dataset: os.DirEntry, file_path, pipeline_info, output_path, output_filename):
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

    # with open(f'kaggle/{output_filename}.json', 'w') as f:
    #     json.dump(nodes, f)
    #
    # with open(f'kaggle/{output_filename}-files.json', 'w') as f:
    #     json.dump(file_elements, f)
    if not os.path.isdir(os.path.join(output_path, output_filename)):
        os.mkdir(os.path.join(output_path, output_filename))

    with open(os.path.join(output_path, output_filename + '.ttl'), 'w') as f:
        f.write(build_pipeline_rdf_page(nodes, file_elements))

    pipeline_info['uri'] = util.create_pipeline_uri(SOURCE, DATASET_NAME, output_filename)
    pipeline_info['dataset'] = util.create_dataset_uri(SOURCE, DATASET_NAME)
    default_graph.append(pipeline_info)


if __name__ == "__main__":
    main()
