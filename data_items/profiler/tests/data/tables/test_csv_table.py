import sys

sys.path.insert(0, '../../../src')

from data.tables.csv_table import CSVTable


def test_csv_table():
    csvTable = CSVTable('table1.csv', 'dataset1', '/data/dataset1', 'origin1')
    assert (csvTable.get_table_name() == 'table1.csv')
    assert (csvTable.get_dataset_name() == 'dataset1')
    assert (csvTable.get_table_path() == '/data/dataset1/table1.csv')
    assert (csvTable.get_origin() == 'origin1')
