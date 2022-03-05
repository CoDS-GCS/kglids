import sys

sys.path.insert(0, '../../../../src')
from analysis.interpreter.interpreter import Interpreter
from analysis.profile_creator.analysers.textual_analyser import TextualAnalyser
from data.tables.csv_table import CSVTable


def test_analyse_df():
    table = CSVTable('countries.csv', 'countries', '../../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    textual_columns_df = interpreter.get_textual_columns()
    textual_analyzer = TextualAnalyser(textual_columns_df)
    textual_analyzer.analyse_columns()
    profiles_info = textual_analyzer.get_profiles_info()
    assert (len(profiles_info) == 2)
    print(profiles_info['country'].keys())
    assert (len(profiles_info['country'].keys()) == 4)
    assert (len(profiles_info['capital'].keys()) == 4)
    assert (len(profiles_info['country']['minhash']) == 512)
    assert (len(profiles_info['capital']['minhash']) == 512)
    assert (profiles_info['capital']['missing_values_count'] == 1)
    assert (profiles_info['country']['missing_values_count'] == 0)
