import sys

sys.path.insert(0, '../../../src')

from analysis.profile_creator.profile_creator import ProfileCreator
from analysis.interpreter.interpreter import Interpreter
from data.tables.csv_table import CSVTable


def test_create_numerical_profiles():
    table = CSVTable('countries.csv', 'countries', '../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    numerical_columns_df = interpreter.get_numerical_columns()
    profile_creator = ProfileCreator(table)
    profiles = list(profile_creator.create_numerical_profiles(numerical_columns_df))
    assert (len(profiles) == 2)

    # check the first profile
    profile1 = profiles[0]
    assert (profile1.get_minhash() == [])
    assert (profile1.get_total_values() == 6)
    assert (profile1.get_distinct_values_count() == 6)
    assert (profile1.get_table_name() == 'countries.csv')
    assert (profile1.get_column_name() == 'population')
    assert (profile1.get_origin() == 'test')
    assert (profile1.get_min_value() == 1701572)
    assert (profile1.get_missing_values_count() == 1)

    # check the second profile
    profile2 = profiles[1]
    assert (profile2.get_minhash() == [])
    assert (profile2.get_total_values() == 6)
    assert (profile2.get_distinct_values_count() == 5)
    assert (profile2.get_table_name() == 'countries.csv')
    assert (profile2.get_column_name() == '% water')
    assert (profile2.get_origin() == 'test')
    assert (profile2.get_min_value() == 0)
    assert (profile2.get_missing_values_count() == 1)


def test_create_textual_profiles():
    table = CSVTable('countries.csv', 'countries', '../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    textual_columns_df = interpreter.get_textual_columns()
    profile_creator = ProfileCreator(table)
    profiles = list(profile_creator.create_textual_profiles(textual_columns_df))
    assert (len(profiles) == 2)

    # check the first profile
    profile1 = profiles[0]
    assert (len(profile1.get_minhash()) == 512)
    assert (profile1.get_total_values() == 7)
    assert (profile1.get_distinct_values_count() == 7)
    assert (profile1.get_table_name() == 'countries.csv')
    assert (profile1.get_column_name() == 'country')
    assert (profile1.get_origin() == 'test')
    assert (profile1.get_min_value() == -1)
    assert (profile1.get_missing_values_count() == 0)

    # check the second profile
    profile2 = profiles[1]
    assert (len(profile2.get_minhash()) == 512)
    assert (profile2.get_total_values() == 7)
    assert (profile2.get_distinct_values_count() == 7)
    assert (profile2.get_table_name() == 'countries.csv')
    assert (profile2.get_column_name() == 'capital')
    assert (profile2.get_origin() == 'test')
    assert (profile2.get_min_value() == -1)
    assert (profile2.get_missing_values_count() == 1)
