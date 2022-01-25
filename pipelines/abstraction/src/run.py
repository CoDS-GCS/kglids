import ast
import os
import pandas as pd
import requests
import time
import re
from src.adapters.repository import Neo4JRepository
from src.pipeline_abstraction import NodeVisitor
from src.datatypes import GraphInformation

overall_start = time.time()

files: os.DirEntry
f = []
# for directory in os.scandir('../kaggle'):
directory = "titanic"
for files in os.scandir(f"../files/{directory}"):
    filename = files.name.split('.')[0]
    if filename != '':
        f.append({
            'file': filename,
            'dataset_name': directory
        })

working_file = {}
table: os.DirEntry
for table in os.scandir(f'../files/data/'):
    if table.name != '.DS_Store':
        try:
            working_file[table.name] = pd.read_csv(table.path)
        except Exception as e:
            print(table.name)


for file in f:
    file, dataset_name = file.values()

    SOURCE = 'kaggle'
    DATASET_NAME = dataset_name
    PYTHON_FILE_NAME = f'{file}.py'

    # file_name = f"{DATASET_NAME}/notebooks/{file}"
    file_name = f"{DATASET_NAME}/{file}"
    with open(f'../files/{file_name}.py', 'r') as src_file:
        src = src_file.read()

    tree = ""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        with open(f'./errors.csv', 'a') as output_file:
            output_file.write(f'{PYTHON_FILE_NAME},{e.msg}\n')

    graph = GraphInformation(PYTHON_FILE_NAME, SOURCE, DATASET_NAME)
    nodeVisitor = NodeVisitor(graph_information=graph)
    nodeVisitor.working_file = working_file

    starting_time = time.time()
    nodeVisitor.visit(tree)
    ending_time = time.time()
    print("Processing time: ", ending_time - starting_time, 'seconds')
    file_elements = [el.str() for el in graph.files.values()]
    libraries = [library.str() for library in graph.libraries.values()]
    nodes = []
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
    neo4j_repository.create_file_nodes(file_elements)
    neo4j_repository.create_library_nodes(libraries)
    if len(nodes) > 2:
        neo4j_repository.create_first_node(nodes[0])
        neo4j_repository.create_node_nodes(nodes[1:])
    neo4j_end = time.time()
    print("Neo4j time:", neo4j_end - neo4j_start, 'seconds')
    n10s_start = time.time()
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

    # unique_edge += re.findall(r'(pipeline:(?!loop|conditional|userDefinedFunction)\w+)', rdf_file)
    # unique_node += re.findall(
    #     r'((?<!@prefix pipeline: )<http://kglids.org/.*>)|(pipeline:(?=loop|conditional|userDefinedFunction)\w+)'
    #     r'|(".*")',
    #     rdf_file
    # )
    # triples_count += len(re.findall(r'((?<!@.{45}> )\.\n)|(;\n)', rdf_file)) - 1

    with open(f'../kaggle_rdf/{file}.ttl', 'w') as output_file:
        output_file.write(rdf_file)

    n10s_end = time.time()
    with open(f'./statistics_{dataset_name}.csv', 'a') as output_file:
        output_file.write(f'{PYTHON_FILE_NAME},{n10s_end - starting_time}\n')
    print("Neosemantic processing:", n10s_end - n10s_start, "seconds")
    print("Total time:", n10s_end - starting_time, "seconds")

overall_end = time.time()
print("Overall time:", overall_end - overall_start, "seconds")
