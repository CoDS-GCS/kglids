from typing import Dict, Optional

import kg_governor.pipeline_abstraction.Calls as Calls
import kg_governor.pipeline_abstraction.util as util
from kg_governor.pipeline_abstraction.Calls import packages
from kg_governor.pipeline_abstraction.util import parse_line_text, create_import_uri, create_file_uri, create_column_name, extract_library_dependencies, create_built_in_uri


class Node:
    __slots__ = ('previous', 'next', 'text', 'uri', 'data_flow',
                 'parameters', 'calls', 'read', 'control_flow')

    def __init__(self, previous, text):
        self.previous = previous
        self.next = None
        self.text = text
        self.uri = ""
        self.control_flow = set()
        self.parameters = []
        self.calls = []
        self.read = []
        self.data_flow = []

    def generate_uri(self, source, dataset, file, node_id):
        self.uri = util.create_statement_uri(source, dataset, file, node_id)

    def str(self):
        control_flows = [flow for flow in self.control_flow]
        parameters = [value.str() for value in self.parameters]
        call_array = [value.repr() for value in self.calls]
        read_array = [value.repr() for value in self.read]
        data_flow = [value.uri for value in self.data_flow]

        return {"uri": self.uri,
                "previous": self.previous.uri if self.previous is not None else None,
                "next": self.next.uri if self.next is not None else None,
                "text": self.text,
                "control_flow": control_flows,
                "parameters": parameters,
                "calls": call_array,
                "read": read_array,
                "dataFlow": data_flow
                }


class AttrEdge:
    __slots__ = ('parameter', 'parameter_value')

    def __init__(self, parameter, parameter_value):
        self.parameter = parameter
        self.parameter_value = parameter_value

    def str(self):
        return {'parameter': self.parameter, 'parameter_value': self.parameter_value}


call_types = {
    Calls.CallType.FUNCTION.value: 'callsFunction',
    Calls.CallType.CLASS.value: 'callsClass',
    Calls.CallType.PACKAGE.value: 'callsPackage',
    Calls.CallType.LIBRARY.value: 'callsLibrary',
    Calls.CallType.NONE.value: 'callsAPI'
}


def get_call_type(library_type: str) -> str:
    return call_types.get(library_type)


class Library:
    __slots__ = ('uri', 'contain', 'type', 'call_type')

    def __init__(self, uri, library_type=None):
        self.uri = uri
        self.contain = dict()
        self.type = library_type
        self.call_type = get_call_type(library_type)

    def str(self):
        libraries = [value.str() for value in self.contain.values()]
        return {'uri': self.uri, 'contain': libraries, 'type': self.type}

    def repr(self):
        return {'uri': self.uri, 'call_type': self.call_type}


class File:
    __slots__ = ('uri', 'contain')

    def __init__(self, uri):
        self.uri = uri
        self.contain = set()

    def __cmp__(self, other):
        return self.uri == other.uri

    def str(self):
        columns = [value.str() for value in self.contain]
        return {'uri': self.uri, 'contain': columns}

    def repr(self):
        return {'uri': self.uri, 'type': 'readsTable'}


class Column:
    __slots__ = 'uri'

    def __init__(self, uri):
        self.uri = uri

    def __cmp__(self, other):
        return self.uri == other.uri

    def str(self):
        return {'uri': self.uri}

    def repr(self):
        return {'uri': self.uri, 'type': 'readsColumn'}


class GraphInformation:
    head: Optional[Node]
    tail: Optional[Node]
    files: Dict[str, File]
    libraries: Dict[str, Library]

    __slots__ = ('head', 'tail', 'files', 'libraries',
                 'python_file_name', 'source', 'dataset_name')

    def __init__(self, python_file_name: str, source: str, dataset_name: str, libraries=None):
        if libraries is None:
            libraries = dict()
        self.head = None
        self.tail = None
        self.files = dict()
        self.libraries = libraries
        self.python_file_name = python_file_name
        self.source = source
        self.dataset_name = dataset_name

    def add_node(self, text):
        if text.startswith('"""'):
            return

        node = Node(self.tail, parse_line_text(text))
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def add_control_flow(self, control_flow):
        for flow in control_flow:
            self.tail.control_flow.add(flow.value)

    def add_file(self, file):
        file_uri = create_file_uri(self.source, self.dataset_name, file.filename)
        file_node = File(file_uri)
        self.tail.read.append(file_node)
        self.files[file.filename] = file_node

    def add_columns(self, filename, columns):
        for column_name in columns:
            column_uri = create_column_name(self.source, self.dataset_name, filename, column_name)
            column = Column(column_uri)
            self.tail.read.append(column)
            self.files.get(filename).contain.add(column)

    def add_import_node(self, path, func=create_import_uri):
        dependencies = extract_library_dependencies(path)
        container = self.libraries
        for i in range(len(dependencies)):
            library_path = dependencies[i]
            lib_uri = func(library_path)
            if '*' in lib_uri:
                continue

            if lib_uri not in container.keys():
                default_call = Calls.Call() if '.' in library_path else Calls.Call(call_type=Calls.CallType.LIBRARY)
                library = Library(lib_uri, packages.get(library_path, default_call).call_type.value)
                container[lib_uri] = library
            else:
                library = container.get(lib_uri)
            if i == len(dependencies) - 1:
                self.tail.calls.append(library)

            container = library.contain

    def add_import_from_node(self, path, name):
        full_path = f'{path}.{name}'
        self.add_import_node(full_path)

    def add_package_call(self, package):
        if package is None:
            return
        self.add_import_node(package)

    def rewrite_node_flow(self):
        current_node = self.tail
        parent_node = self.tail.previous
        ancestor_node = parent_node.previous

        if parent_node == self.head:
            self.head = current_node

        temp = current_node.next
        current_node.next = parent_node
        parent_node.next = temp
        if ancestor_node is not None:
            ancestor_node.next = current_node

        current_node.previous = ancestor_node
        parent_node.previous = current_node

        self.tail = parent_node

    def insert_before(self, node: Node):
        current_node = self.tail
        current_node.previous.next = None
        self.tail = current_node.previous

        if node == self.head:
            self.head = current_node

        current_node.previous = node.previous
        node.previous = current_node
        current_node.next = node
        if current_node.previous is not None:
            current_node.previous.next = current_node

    def add_parameter(self, parameter, value):
        edge = AttrEdge(parameter, str(value))
        self.tail.parameters.append(edge)

    def add_parameters(self, parameters: dict):
        for key, value in parameters.items():
            edge = AttrEdge(key, str(value))
            self.tail.parameters.append(edge)

    def add_built_in_call(self, library):
        self.add_import_node(library, create_built_in_uri)

    def add_data_flows(self, node: Node):
        node.data_flow.append(self.tail)

    def add_concurrent_flow(self, node: Node or None):
        if node is None:
            self.tail.previous.data_flow.append(self.tail)
        else:
            node.previous.data_flow.append(self.tail)
