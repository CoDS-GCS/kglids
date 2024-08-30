import ast
import errno
from collections import ChainMap
from datetime import datetime
import json
import multiprocessing as mp
import os
import pandas as pd
from glob import glob
from pathlib import Path

import src.Calls as Calls
import src.util as util
from typing import Dict
from tqdm import tqdm

from src.pipeline_abstraction import NodeVisitor
from src.datatypes import GraphInformation
from src.json_to_rdf import build_pipeline_rdf_page, build_default_rdf_page, build_library_rdf_page
from src.config import abstraction_config
from src.util import url_encode

def main():
    start_time = datetime.now()

    pipelines = []
    datasets = list(os.scandir(abstraction_config.data_source_path))

    # loop through datasets & pipelines
    for dataset in tqdm(datasets):
        pipeline: os.DirEntry
        if dataset.is_dir():
            working_file = {}
            table: os.DirEntry
            tables = glob(os.path.join(dataset.path, '**', '*.csv'), recursive=True)

            for table in tables:
                try:
                    try:
                        working_file[Path(table).name] = pd.read_csv(table, nrows=1)
                    except:
                        working_file[Path(table).name] = pd.read_csv(table, nrows=1,
                                                                     engine='python', encoding_errors='replace')
                except Exception as e:
                    print("-<>", table, e)

            if not os.path.isdir(f'{dataset.path}/notebooks/'):
                continue
            for pipeline in os.scandir(f'{dataset.path}/notebooks'):
                if pipeline.is_dir():
                    try:
                        with open(f'{pipeline.path}/pipeline_info.json', 'r') as f:
                            pipeline_info = json.load(f)

                        file: os.DirEntry
                        for file in os.scandir(pipeline.path):
                            if '.py' in file.name:
                                pipelines.append((working_file, dataset.name, file.path, pipeline_info,
                                                  abstraction_config.output_graphs_path,
                                                  pipeline_info['id']))
                    except FileNotFoundError as e:
                        continue

    print(datetime.now(), ': Analyzing', len(pipelines), 'pipelines for', len(datasets), 'datasets.')
    pool = mp.Pool()
    default_and_library_graphs = list(tqdm(pool.imap_unordered(pipeline_analysis, pipelines), total=len(pipelines)))

    # combine the default and library graphs from individual pipelines
    default_graph = [item[0] for item in default_and_library_graphs if item]
    libraries = ChainMap(*[item[1] for item in default_and_library_graphs if item])
    libs = [library.str() for library in libraries.values()]

    with open(os.path.join(abstraction_config.output_graphs_path, 'library.ttl'), 'w') as f:
        f.write(build_library_rdf_page(libs))

    with open(os.path.join(abstraction_config.output_graphs_path, 'default.ttl'), 'w') as f:
        f.write(build_default_rdf_page(default_graph))

    print(datetime.now(), ': Done. Total Time:', datetime.now() - start_time)
    print(Calls.read_csv_call.count)


def pipeline_analysis(args):
    working_file, dataset, file_path, pipeline_info, output_path, output_filename = args

    SOURCE = abstraction_config.data_source
    DATASET_NAME = dataset
    PYTHON_FILE_NAME = output_filename[:200] # trim filename to 200 characters to avoid OSError

    # Read pipeline file
    with open(file_path, 'r') as src_file:
        src = src_file.read()

    try:
        tree = ast.parse(src)
    except Exception as e:
        # with open(f'./errors.csv', 'a') as output_file:
        #     output_file.write(f'{PYTHON_FILE_NAME},{e}\n')
        return

    # Initialize graph information linked list and Node Visitor
    graph = GraphInformation(PYTHON_FILE_NAME, SOURCE, DATASET_NAME, libraries=None)
    node_visitor = NodeVisitor(graph_information=graph)
    node_visitor.working_file = working_file

    # Pipeline analysis
    try:
        node_visitor.visit(tree)
    except Exception as e:
        print('Error parsing notebook:', file_path)
        # with open(f'./errors.csv', 'a') as output_file:
        #     output_file.write(f'{PYTHON_FILE_NAME},{e}\n')
        return

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

    os.makedirs(os.path.join(output_path, DATASET_NAME), exist_ok=True)

    with open(os.path.join(output_path, DATASET_NAME, url_encode(PYTHON_FILE_NAME) + '.ttl'), 'w') as f:
        f.write(build_pipeline_rdf_page(nodes, file_elements))

    pipeline_info['uri'] = util.create_pipeline_uri(SOURCE, DATASET_NAME, PYTHON_FILE_NAME)
    pipeline_info['dataset'] = util.create_dataset_uri(SOURCE, DATASET_NAME)

    return pipeline_info, graph.libraries


if __name__ == "__main__":
    main()
