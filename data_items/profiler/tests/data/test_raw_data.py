import sys

sys.path.insert(0, '../../src')
print(sys.path)
from data.raw_data import RawData


def test_raw_data():
    rawData = RawData(123456789, 'demo', 'dataset1', 'usr/src/data', 'table1',
                      'column1', ['value1', 'value2', 'value3'])

    assert (rawData.get_rid() == 123456789)
    assert (rawData.get_origin() == 'demo')
    assert (rawData.get_path() == 'usr/src/data')
    assert (rawData.get_dataset_name() == 'dataset1')
    assert (rawData.get_table_name() == 'table1')
    assert (rawData.get_column_name() == 'column1')
    assert (rawData.get_values() == ['value1', 'value2', 'value3'])
