import unittest
import util

from src.datatypes import Node, AttrEdge, Library, File, Column


class DataTypes(unittest.TestCase):
    def test_Node_str_output(self):
        previous_uri = util.create_statement_uri("kagle", "titanic", "file.py", 1)
        node_uri = util.create_statement_uri("kagle", "titanic", "file.py", 1)
        text = "random text"
        control_flow = util.ControlFlow.CONDITIONAL.value
        parameter = "label"
        parameter_value = 5
        edge = AttrEdge(parameter, parameter_value)
        lib1_uri = util.create_import_uri('pandas')
        lib2_uri = util.create_import_uri('pandas.DataFrame')
        pandas = Library(lib1_uri)
        dataframe = Library(lib2_uri)
        col_uri = util.create_column_name("kaggle", "titanic", "train.csv", "Age")
        col_uri2 = util.create_column_name("kaggle", "titanic", "train.csv", "Sex")
        column = Column(col_uri)
        column2 = Column(col_uri2)

        previous_node = Node(None, "", previous_uri)
        node = Node(previous_node, text, node_uri)
        node.control_flow.add(control_flow)
        node.parameters.append(edge)
        node.calls.append(pandas)
        node.calls.append(dataframe)
        node.read.append(column)
        node.read.append(column2)

        expected = {"uri": node_uri,
                    "previous": previous_uri,
                    "text": text,
                    "control_flow": [control_flow],
                    "parameters": [{"parameter": parameter, "parameter_value": parameter_value}],
                    "calls": [{"uri": lib1_uri}, {"uri": lib2_uri}],
                    "read": [{"uri": col_uri}, {"uri": col_uri2}]
                    }

        self.assertEqual(expected, node.str())

    def test_Column_str_output(self):
        col_uri = util.create_column_name("kaggle", "titanic", "train.csv", "Age")
        column = Column(col_uri)
        expected = {'uri': col_uri}

        self.assertEqual(expected, column.str())

    def test_Library_str_output(self):
        lib1_uri = util.create_import_uri('pandas')
        lib2_uri = util.create_import_uri('pandas.DataFrame')
        lib3_uri = util.create_import_uri('pandas.read_csv')

        pandas = Library(lib1_uri)
        dataframe = Library(lib2_uri)
        read_csv = Library(lib3_uri)

        pandas.contain[lib2_uri] = dataframe
        pandas.contain[lib3_uri] = read_csv
        expected = {'uri': lib1_uri, 'contain': [
            {'uri': lib2_uri, 'contain': []},
            {'uri': lib3_uri, 'contain': []}
        ]}

        self.assertEqual(expected, pandas.str())

    def test_AttrEdge_str_output(self):
        parameter = "label"
        parameter_value = 5
        edge = AttrEdge(parameter, parameter_value)
        expected = {'parameter': parameter, 'parameter_value': parameter_value}
        self.assertEqual(expected, edge.str())

    def test_File_str_output(self):
        table_uri = util.create_file_uri("kaggle", "titanic", "train.csv")
        col_uri = util.create_column_name("kaggle", "titanic", "train.csv", "Age")
        column = Column(col_uri)
        table = File(table_uri)
        table.contain.add(column)
        expected = {'uri': table_uri, 'contain': [
            {'uri': col_uri}
        ]}

        self.assertEqual(expected, table.str())
