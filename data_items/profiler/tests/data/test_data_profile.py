import sys

sys.path.insert(0, '../../src')
from data.column_profile import ColumnProfile


def test_data_profile():
    profile = ColumnProfile(123456789, 'demo', 'dataset1', 'usr/src/data', 'table1',
                          'column1', 'type1', 1000, 50, 9, 0, 10, 5, 4, 3, [2021, 2020, 2019])

    assert (profile.get_pid() == 123456789)
    assert (profile.get_origin() == 'demo')
    assert (profile.get_path() == 'usr/src/data')
    assert (profile.get_dataset_name() == 'dataset1')
    assert (profile.get_table_name() == 'table1')
    assert (profile.get_column_name() == 'column1')
    assert (profile.get_total_values() == 1000)
    assert (profile.get_distinct_values_count() == 50)
    assert (profile.get_missing_values_count() == 9)
    assert (profile.get_data_type() == 'type1')
    assert (profile.get_min_value() == 0)
    assert (profile.get_max_value() == 10)
    assert (profile.get_mean() == 5)
    assert (profile.get_median() == 4)
    assert (profile.get_iqr() == 3)
    assert (profile.get_minhash() == [2021, 2020, 2019])
