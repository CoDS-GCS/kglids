import sys

sys.path.insert(0, '../../src')
from storage.elasticsearchDB import ElasticSearchDB
from data.raw_data import RawData
from data.data_profile import DataProfile

client = ElasticSearchDB(host='localhost', port=9200)


def test_data_storing():
    rawData1 = RawData(123456789, 'test', 'dataset1', 'usr/src/data', 'table1',
                       'column1', ['value1', 'value2', 'value3'])
    rawData2 = RawData(987654321, 'test', 'dataset2', 'usr/src/data', 'table2',
                       'column2', ['value11', 'value22', 'value33'])
    rawData = [rawData1, rawData2]
    client.store_data(rawData)
    number_of_documents = client.count_per_index('raw_data')
    client.delete_index('raw_data')
    assert (number_of_documents == 2)


def test_profile_storing():
    profile1 = DataProfile(123456789, 'test', 'dataset1', 'usr/src/data', 'table1',
                           'column1', 'type1', 1000, 50, 9, 0, 10, 5, 4, 3, [2021, 2020, 2019])
    profile2 = DataProfile(987654321, 'test', 'dataset2', 'usr/src/data', 'table2',
                           'column2', 'type2', 1000, 50, 9, 0, 10, 5, 4, 3, [2022, 2002, 2091])

    profiles = [profile1, profile2]
    client.store_profiles(profiles)
    number_of_documents = client.count_per_index('profiles')
    client.delete_index('profiles')
    assert (number_of_documents == 2)
