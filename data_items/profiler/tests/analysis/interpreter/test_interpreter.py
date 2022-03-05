import sys

sys.path.insert(0, '../../../src')

from data.tables.csv_table import CSVTable
from analysis.interpreter.interpreter import Interpreter


def test_determine_data_types():
    table = CSVTable('countries.csv', 'countries', '../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    col_types = sorted(interpreter.get_columns_types())
    print(col_types)
    assert (len(col_types) == 4)
    assert (col_types[0][1] == 'double')
    assert (col_types[1][1] == 'string')
    assert (col_types[2][1] == 'string')
    assert (col_types[3][1] == 'int')


def test_getters():
    table = CSVTable('countries.csv', 'countries', '../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    # Get the numerical cols

    numerical_column_names = sorted(interpreter.get_numerical_column_names())
    assert (len(numerical_column_names) == 2)
    assert (numerical_column_names[0] == '`% water`')
    assert (numerical_column_names[1] == '`population`')

    # Get the textual cols

    textual_column_names = sorted(interpreter.get_textual_column_names())
    assert (len(textual_column_names) == 2)
    assert (textual_column_names[0] == '`capital`')
    assert (textual_column_names[1] == '`country`')

    # Check that the cast to numerical is correct
    numerical_types = sorted(interpreter.get_numerical_columns().dtypes)
    assert (numerical_types[0][0] == '% water')
    assert (numerical_types[0][1] == 'double')
    assert (numerical_types[1][0] == 'population')
    assert (numerical_types[1][1] == 'int')

    # Check that the content of the textual df contains only textual columns
    textual_types = sorted(interpreter.get_textual_columns().dtypes)
    assert (textual_types[0][0] == 'capital')
    assert (textual_types[0][1] == 'string')
    assert (textual_types[1][0] == 'country')
    assert (textual_types[1][1] == 'string')


def test_get_raw_data():
    table = CSVTable('countries.csv', 'countries', '../../resources/datasets/countries/', 'test')
    interpreter = Interpreter(table)
    raw_data = list(interpreter.get_raw_data())
    assert (len(raw_data) == 4)
