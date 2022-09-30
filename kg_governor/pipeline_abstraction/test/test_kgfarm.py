import ast
import unittest

import pandas as pd

import src.util as util
import src.datatypes as Datatypes
from src.Calls import File, pd_dataframe
from src.datatypes import GraphInformation
from src.pipeline_abstraction import NodeVisitor
from test.test_pipeline_abstraction import Test, parse_and_visit_node

kglids_library = "http://kglids.org/pipeline/library/"
FILENAME = "test.py"
SOURCE = "<SOURCE>"
DATASET_NAME = "<DATASET_NAME>"


def parse_and_visit_node_with_file(lines: str, graph: GraphInformation, filename: str,
                                   columns: list, variable: str = 'df') -> NodeVisitor:
    parse_tree = ast.parse(lines)
    node_visitor = NodeVisitor(graph_information=graph)

    node_visitor.working_file[filename] = pd.DataFrame(columns=columns)
    graph.files[filename] = Datatypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
    node_visitor.files = {variable: File(filename)}
    node_visitor.variables = {variable: pd_dataframe}

    node_visitor.visit(parse_tree)
    return node_visitor


class KGFarmTest(Test):
    def test_when_column_drop_then_it_doesnt_appear_as_feature(self):
        value = "import pandas as pd\n" \
                "from sklearn.preprocessing import OneHotEncoder\n" \
                "df = pd.read_csv('file.csv')\n" \
                "encoder = OneHotEncoder()\n" \
                "df.drop('a', 1, inplace=True)\n" \
                "encoder.fit(df)\n"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c'])
        column_1 = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')
        column_2 = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'c')

        self.assertListEqual([column_1, column_2], self.graph.tail.read)

    def test_when_parameter_are_passed_within_variable_then_the_parameter_are_associated_to_the_function(self):
        value = "import pandas as pd\n" \
                "from sklearn.preprocessing import OneHotEncoder\n" \
                "df = pd.read_csv('file.csv')\n" \
                "encoder = OneHotEncoder()\n" \
                "column = 'a'\n" \
                "df.drop(column, 1, inplace=True)\n" \
                "encoder.fit(df)\n"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c'])
        column_1 = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')
        column_2 = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'c')

        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a'),
            self.graph.tail.previous.read[0].uri
        )
        self.assertListEqual([column_1, column_2], self.graph.tail.read)

    def test_when_constant_passed_to_variable_then_constant_saved_as_variable(self):
        value = "x = 'a'"

        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertIn('x', node_visitor.variables.keys())
        self.assertEqual('a', node_visitor.variables.get('x'))

    def test_when_array_passed_to_variable_then_array_saved_as_variable(self):
        value = "x = ['a', 'b']"

        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertIn('x', node_visitor.variables.keys())
        self.assertListEqual(['a', 'b'], node_visitor.variables.get('x'))

    def test_when_variable_is_save_then_it_assigned_package_to_variable(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a'])

        self.assertEqual(pd_dataframe, node_visitor.variables.get('df'))

    def test_when_variable_assigned_column_then_column_is_read_when_passed_as_index(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')\n" \
                "col = 'b'\n" \
                "df[col] = 1"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['b'])

        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b'),
            self.graph.tail.read[0].uri)

    def test_when_variable_assigned_column_then_column_is_read_when_passed_as_parameter(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')\n" \
                "col = 'b'\n" \
                "df.drop(col, 1, inplace=True)\n"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['b'])

        print([(x.parameter, x.parameter_value) for x in self.graph.tail.parameters])
        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b'),
            self.graph.tail.read[0].uri)

    def test_when_variable_assigned_column_in_array_then_column_is_read_when_passed_as_parameter(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')\n" \
                "col = ['b']\n" \
                "df.drop(col, 1, inplace=True)\n"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['b'])

        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b'),
            self.graph.tail.read[0].uri)


if __name__ == '__main__':
    unittest.main()

