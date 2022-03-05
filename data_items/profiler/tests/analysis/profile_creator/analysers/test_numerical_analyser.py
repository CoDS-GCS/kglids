import sys

sys.path.insert(0, '../../../../src')
from analysis.interpreter.interpreter import Interpreter
from analysis.profile_creator.analysers.numerical_analyser import NumericalAnalyser
from data.tables.csv_table import CSVTable


def test_analyse_df():
    table = CSVTable('countries.csv', 'countries', '../../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    numerical_columns_df = interpreter.get_numerical_columns()
    numerical_analyzer = NumericalAnalyser(numerical_columns_df)
    numerical_analyzer.analyse_columns()
    profiles_info = numerical_analyzer.get_profiles_info()
    info_fields = ['25%', '50%', '75%', 'count', 'distinct_values_count', 'max', 'mean', 'min', 'missing_values_count',
                   'stddev']
    assert (len(profiles_info) == 2)
    assert (len(profiles_info['% water']) == 10)
    assert (len(profiles_info['population']) == 10)
    assert (sorted(profiles_info['% water'].keys()) == info_fields)
    assert (profiles_info['% water']['missing_values_count'] == 1)
    assert (sorted(profiles_info['population'].keys()) == info_fields)
    assert (profiles_info['population']['missing_values_count'] == 1)
