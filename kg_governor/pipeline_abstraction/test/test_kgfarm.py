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

    def test_when_interpolate_then_function_extracted_correctly(self):
        value = "import pandas as pd\n" \
                "df = pd.DataFrame()\n" \
                "df.interpolate(method='linear', limit_direction='forward', axis=0)"

        parse_and_visit_node(value, self.graph)

        lib_uri = util.create_import_from_uri('pandas.DataFrame', 'interpolate')
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)

    def test_when_label_encoder_fit_transform_then_save_it_as_call(self):
        value = "from sklearn.preprocessing import LabelEncoder\n" \
                "import pandas as pd\n" \
                "label = LabelEncoder()\n" \
                "df = pd.DataFrame(columns=['Gender'])\n" \
                "df['Gender'] = label.fit_transform(df['Gender'])"

        parse_and_visit_node(value, self.graph)

        lib_uri = util.create_import_from_uri('sklearn.preprocessing.LabelEncoder', 'fit_transform')
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)

    def test_when_dataframe_called_by_transformation_then_it_reads_the_dataframe_column(self):
        value = "from sklearn.preprocessing import LabelEncoder\n" \
                "import pandas as pd\n" \
                "label = LabelEncoder()\n" \
                "df = pd.read_csv('file.csv')\n" \
                "label.fit_transform(df)"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['age'])
        lib_uri = util.create_import_from_uri('sklearn.preprocessing.LabelEncoder', 'fit_transform')
        column_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'age')

        self.assertEqual(0, len(node_visitor.columns))
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)
        self.assertEqual(1, len(self.graph.tail.read))
        self.assertEqual(column_uri, self.graph.tail.read[0].uri)

    def test_when_column_is_drop_within_transformation_then_it_correctly_linked_to_read_column(self):
        value = "from sklearn.preprocessing import StandardScaler\n" \
                "import pandas as pd\n" \
                "scaler = StandardScaler()\n" \
                "df = pd.read_csv('file.csv')\n" \
                "scaled_df = scaler.fit_transform(df.drop('Outcome', axis=1))"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['Gender', 'Outcome'], 'df')
        lib_uri = util.create_import_from_uri('sklearn.preprocessing.StandardScaler', 'fit_transform')
        column_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'Gender')

        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)
        self.assertEqual(2, len(self.graph.tail.read))
        self.assertEqual(column_uri, self.graph.tail.read[0].uri)

    def test_when_transformation_apply_to_dataframe_then_it_correctly_link_the_call_and_parameter(self):
        value = "encoder = LabelEncoder()\n" \
                "encoded = df[categorical_columns].apply(encoder.fit_transform)"

    def test_when_column_transformer_is_used_then_it_correctly_link_the_call_with_the_feature(self):
        value = "transformers = [\n" \
                "\t('binary', OrdinalEncoder(), binary_columns),\n" \
                "\t('nominal', OneHotEncoder(), nominal_columns),\n" \
                "\t('numerical', StandardScaler(), numerical_columns)\n" \
                "]\n" \
                "transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')"


if __name__ == '__main__':
    unittest.main()

