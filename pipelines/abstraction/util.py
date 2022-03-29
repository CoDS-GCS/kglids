import zlib
import urllib.parse
from enum import Enum

import Calls
from Calls import Elements, packages


def is_file(string):
    if not isinstance(string, str):
        return False
    return '.csv' in string


def generate_id(dataset_name: str, table_name: str, column_name: str):
    return zlib.crc32(bytes(dataset_name + table_name + column_name, 'utf-8'))


def get_package_info(package_name, alias: any):
    if isinstance(package_name, Calls.Call):
        return package_name
    if package_name is None or not isinstance(package_name, str):
        return None
    if '.' in package_name:
        *path, name = package_name.split('.')
        if path[0] in alias.keys():
            path[0] = alias.get(path[0])
        path = '.'.join(path)
        package = None
        for element in Elements:
            if element.name == name and element.library_path == path:
                package = element
        return package
    else:
        for element in Elements:
            if element.name == package_name:
                return element
    return None


def get_package(package_name, alias):
    if isinstance(package_name, Calls.Call):
        return package_name
    if package_name is None or not isinstance(package_name, str):
        return None
    try:
        if '.' in package_name:
            path = package_name.split('.')
            if path[0] in alias.keys():
                path[0] = alias.get(path[0])
            path = '.'.join(path)
            package = packages[path]

            return package
        else:
            return packages[package_name]
    except KeyError:
        return None


def get_path_package_info(path, name):
    package = None
    for element in Elements:
        if element.name == name and element.library_path == path:
            package = element

    return package


def create_file_id(source: str, dataset: str, table_name: str) -> str:
    return f"kglids.org/{url_encode(source)}/{url_encode(dataset)}/" \
           f"dataResource/{url_encode(table_name)}"


def create_column_name(source: str, dataset: str, table_name: str, column: str):
    return f"http://kglids.org/resource/{url_encode(source)}/" \
           f"{url_encode(dataset)}/{url_encode(table_name)}/" \
           f"{url_encode(column)}"


def create_statement_uri(source: str, dataset_name: str, python_file_name: str, line_id: int):
    return f"http://kglids.org/resource/{source}/" \
           f"{url_encode(dataset_name)}/{url_encode(python_file_name)}/" \
           f"s{line_id}"


def create_import_uri(library_name):
    path = library_name.replace('.', '/')
    return f"http://kglids.org/resource/library/{path}"


def create_import_from_uri(path, library_name):
    path = path.replace('.', '/')
    return "http://kglids.org/resource/library/" \
           f"{path}/{library_name}"


def create_built_in_uri(library_name):
    path = library_name.replace('.', '/')
    return "http://kglids.org/resource/library/builtin/" \
           f"{path}"


def create_file_uri(source: str, dataset_name: str, file_name: str):
    return f"http://kglids.org/resource/{url_encode(source)}/" \
           f"{url_encode(dataset_name)}/{url_encode(file_name)}"


def create_dataset_uri(source: str, dataset_name: str):
    return f"http://kglids.org/resource/{url_encode(source)}/" \
           f"{url_encode(dataset_name)}"


def create_source_uri(source: str):
    return f"http://kglids.org/resource/{url_encode(source)}/"


def create_pipeline_uri(source, dataset_name, file_name):
    return f"http://kglids.org/resource/{url_encode(source)}/" \
           f"{url_encode(dataset_name)}/" \
           f"{url_encode(file_name)}"


def url_encode(string):
    return urllib.parse.quote(str(string), safe='')  # safe parameter is important.


def parse_line_text(text: str) -> str:
    return text.replace('\n', '')


def extract_library_dependencies(path):
    path = path.split('.')
    return ['.'.join(path[:i]) for i in range(1, len(path) + 1)]


class ControlFlow(Enum):
    METHOD = "http://kglids.org/resource/userDefinedFunction"
    LOOP = "http://kglids.org/resource/loop"
    CONDITIONAL = "http://kglids.org/resource/conditional"
    IMPORT = "http://kglids.org/resource/import"
