import os
import queue

from data.tables.csv_table import CSVTable
from data.utils.file_type import FileType


def get_file_type(filename: str) -> FileType:
    if filename.endswith('.csv'):
        return FileType.CSV


def create_sources_from_datasets(datasource: str, datasets: list, tables: queue):
    for dataset in datasets:
        for filename in os.listdir(dataset['path']):
            if get_file_type(filename) == FileType.CSV:
                csvTable = CSVTable(datasource, filename, dataset['name'], dataset['path'], dataset['origin'])
                tables.put(csvTable)
