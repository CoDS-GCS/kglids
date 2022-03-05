import sys

sys.path.insert(0, '../../src')
from storage.utils import serialize_profiles, serialize_rawData
from data.raw_data import RawData
from data.data_profile import DataProfile


def test_profile_serialization():
    profile = DataProfile(123456789, 'test', 'dataset1', 'usr/src/data', 'table1',
                          'column1', 'type1', 1000, 50, 9, 0, 10, 5, 4, 3, [2021, 2020, 2019])
    serializedProfile = serialize_profiles([profile])
    actual_serialization = {'_index': 'profiles',
                            '_source': {'id': 123456789, 'origin': 'test', 'datasetName': 'dataset1',
                                        'path': 'usr/src/data', 'tableName': 'table1', 'columnName': 'column1',
                                        'dataType': 'type1', 'totalValuesCount': 1000, 'distinctValuesCount': 50,
                                        'missingValuesCount': 9, 'minValue': 0, 'maxValue': 10, 'avgValue': 5,
                                        'median': 4, 'iqr': 3,
                                        'minhash': '[2021, 2020, 2019]'}}

    assert (list(serializedProfile)[0] == actual_serialization)


def test_rawData_serialization():
    rawData = rawData = RawData(123456789, 'test', 'dataset1', 'usr/src/data', 'table1',
                                'column1', ['value1', 'value2', 'value3'])
    serializedRawData = serialize_rawData([rawData])
    assert (list(serializedRawData)[0] == {'_index': 'raw_data',
                                           '_source': {'id': 123456789, 'origin': 'test', 'datasetName': 'dataset1',
                                                       'path': 'usr/src/data', 'tableName': 'table1',
                                                       'columnName': 'column1',
                                                       'values': '["value1", "value2", "value3"]'}})
