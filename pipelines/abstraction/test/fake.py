from src.adapters.repository import AbstractRepository
from util import (ControlFlow, create_column_name, create_statement_uri, create_import_uri,
                  create_import_from_uri, create_file_uri)


class Node:
    uri: str
    body: str
    label: str

    def __init__(self, uri: str = '', body: str = '', label: str = ''):
        self.uri = uri
        self.body = body
        self.label = label


class Edge:
    id: str
    start: Node
    end: Node
    label: str

    def __init__(self, id: str = '', start: Node = None, end: Node = None, label: str = ''):
        self.id = id
        self.start = start
        self.end = end
        self.label = label


def create_edge_id(start, end):
    return f"{start}-{end}"


class FakeRepository(AbstractRepository):
    def __init__(self, python_file_name, source, dataset_name):
        self.result = {}
        self.python_file_name = python_file_name
        self.source = source
        self.dataset_name = dataset_name

    def create_control_flow_node(self):
        self.result['Conditional'] = Node(uri=ControlFlow.CONDITIONAL.value, label='Conditional')
        self.result['Loop'] = Node(uri=ControlFlow.LOOP.value, label='Loop')
        self.result['Method'] = Node(uri=ControlFlow.METHOD.value, label='Method')

    def create_line_node(self, line_id, predecessor, text):
        node_uri = create_statement_uri(self.source, self.dataset_name, self.python_file_name, line_id)
        text_id = f"text{line_id}"
        self.result[line_id] = Node(uri=node_uri)
        self.result[text_id] = Node(body=text)
        edge_id = create_edge_id(line_id, text_id)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=self.result.get(line_id),
                                    end=self.result.get(text_id),
                                    label="hasText")
        if predecessor is not None:
            l1 = self.result.get(predecessor)
            l2 = self.result.get(line_id)
            edge_id = create_edge_id(predecessor, line_id)
            self.result[edge_id] = Edge(id=edge_id,
                                        start=l1,
                                        end=l2,
                                        label='NEXT')

    def create_control_flow_edge(self, node_id, control_flow):
        for flow in control_flow:
            line = self.result.get(node_id)
            cf = self.result.get(flow)
            edge_id = create_edge_id(node_id, flow.value)
            self.result[edge_id] = Edge(id=edge_id,
                                        start=line,
                                        end=cf,
                                        label=flow)

    def create_import_node(self, node_id, name):
        statement_node = self.result.get(node_id)
        lib_uri = create_import_uri(name)
        import_node = Node(uri=lib_uri)
        self.result[lib_uri] = import_node
        edge_id = create_edge_id(node_id, lib_uri)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=statement_node,
                                    end=import_node,
                                    label='CALLS')

    def create_import_from_node_with_edge(self, node_id, path, library_name):
        statement_node = self.result.get(node_id)
        lib_uri = create_import_from_uri(path, library_name)
        import_node = Node(uri=lib_uri)
        self.result[lib_uri] = import_node
        edge_id = create_edge_id(node_id, lib_uri)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=statement_node,
                                    end=import_node,
                                    label='CALLS')

    def create_package_call_edge(self, node_id, package):
        if package is None:
            return
        node_uri = create_statement_uri(self.source, self.dataset_name, self.python_file_name, node_id)
        lib_uri = create_import_uri(package)

        statement_node = self.result.get(node_id)
        lib_node = self.result.get(lib_uri)
        edge_id = create_edge_id(node_id, lib_uri)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=statement_node,
                                    end=lib_node,
                                    label='CALLS')

    def create_file_with_edge(self, node_id, file):
        node = self.result.get(node_id)
        file_uri = create_file_uri(self.source, self.dataset_name, file.filename)
        self.result[file.filename] = Node(uri=file_uri)
        edge_id = create_edge_id(node_id, file.filename)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=node,
                                    end=self.result.get(file.filename),
                                    label='OPEN')

    def create_column_connection_edge(self, node_id, file, source, dataset, table_name, columns):
        columns = [create_column_name(source, dataset, table_name, col)
                   for col in columns if col is not None]
        node = self.result.get(node_id)
        file_node = self.result.get(file.filename)
        for column in columns:
            if column not in self.result.keys():
                self.result[column] = Node(uri=column)
            file_edge_id = create_edge_id(file.filename, column)
            self.result[file_edge_id] = Edge(id=file_edge_id,
                                             end=file_node,
                                             start=self.result[column],
                                             label='IS_PART_OF')
            node_edge_id = create_edge_id(node_id, column)
            self.result[node_edge_id] = Edge(id=node_edge_id,
                                             start=node,
                                             end=self.result[column],
                                             label='MANIPULATE')

    def add_parameter_node_with_edge(self, node_id, parameter, value):
        statement_node = self.result.get(node_id)
        literal = Node(body=parameter)
        literal_id = f"{node_id}{parameter}"
        self.result[literal_id] = literal
        edge_id = create_edge_id(node_id, parameter)
        self.result[edge_id] = Edge(id=edge_id,
                                    start=statement_node,
                                    end=literal,
                                    label=f"hasParameter-{value}")

    def rewrite_node_flow(self, parent_node, current_node):
        edge1_id = [key for key in self.result.keys()
                    if f"-{parent_node}" in str(key)
                    and self.result.get(key).label == 'NEXT'][0]
        edge1 = self.result.get(edge1_id)
        edge_start_1 = edge1_id.split('-')[0]
        edge2_id = create_edge_id(parent_node, current_node)
        edge2 = self.result.get(edge2_id)
        new_edge_1_id = create_edge_id(edge_start_1, current_node)
        self.result[new_edge_1_id] = Edge(id=new_edge_1_id,
                                          start=edge1.start,
                                          end=edge2.end,
                                          label='NEXT')
        new_edge_2_id = create_edge_id(current_node, parent_node)
        self.result[new_edge_2_id] = Edge(id=new_edge_1_id,
                                          start=edge2.end,
                                          end=edge2.start,
                                          label='NEXT')
        self.result.pop(edge1_id)
        self.result.pop(edge2_id)

    def remove_node_and_elements(self, node_id, predecessor):
        text_id = f"text{node_id}"
        self.result.pop(node_id)
        self.result.pop(text_id)
        edge_id = create_edge_id(node_id, text_id)
        self.result.pop(edge_id)

        if predecessor is not None:
            edge_id = create_edge_id(predecessor, node_id)
            if edge_id in self.result.keys():
                self.result.pop(edge_id)

    def clean_up(self):
        pass  # Not Implemented

