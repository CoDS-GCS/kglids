import queue
import sys

sys.path.insert(0, '../../src')

from data.utils.yaml_parser import YamlParser
from orchestration.utils import get_file_type, create_sources_from_datasets
from data.utils.file_type import FileType


def test_get_file_type():
    assert (get_file_type('countries.csv') == FileType.CSV)
    assert (get_file_type('countries.ods') != FileType.CSV)


def test_create_sources_from_datasets():
    yamlParser = YamlParser('../resources/config.yml')
    yamlParser.process_config_file()
    datasets = yamlParser.get_datasets_info()
    tables = queue.Queue()
    create_sources_from_datasets(datasets, tables)
    assert (tables.qsize() == 2)
    tables_list = []
    for _ in range(tables.qsize()):
        tables_list.append(tables.get().get_table_name())
    assert (sorted(tables_list) == ['countries.csv', 'players.csv'])
