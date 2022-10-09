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
    graph.files[filename] = Datatypes.File(util.create_file_uri(SOURCE, DATASET_NAME, filename))
    node_visitor.files = {variable: File(filename)}
    node_visitor.variables = {variable: pd_dataframe}
    node_visitor.var_columns[variable] = columns

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

        self.assertListEqual([column_1, column_2], [x.uri for x in self.graph.tail.read])

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
        self.assertListEqual([column_1, column_2], [x.uri for x in self.graph.tail.read])

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

    def test_when_transforming_a_single_column_then_read_only_one_column(self):
        value = "from sklearn.preprocessing import LabelEncoder\n" \
                "import pandas as pd\n" \
                "label = LabelEncoder()\n" \
                "df = pd.read_csv('file.csv')\n" \
                "df['Gender'] = label.fit_transform(df['Gender'])"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['Gender', 'a', 'b'], 'df')

        col_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'Gender')
        self.assertEqual(2, len(self.graph.tail.read))
        self.assertEqual(col_uri, self.graph.tail.read[0].uri)
        self.assertEqual(col_uri, self.graph.tail.read[1].uri)

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

    def test_when_separating_dataframe_then_correct_column_in_slice_are_saved_for_the_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0:2]"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'], 'df')
        col_uri_a = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a')
        col_uri_b = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')

        self.assertEqual(2, len(self.graph.tail.read))
        self.assertEqual(col_uri_a, self.graph.tail.read[0].uri)
        self.assertEqual(col_uri_b, self.graph.tail.read[1].uri)
        self.assertListEqual(['a', 'b'], node_visitor.var_columns.get('X'))
        self.assertIn('X', node_visitor.files.keys())

    def test_when_separating_dataframe_then_correct_column_index_is_saved_for_the_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0]"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b'], 'df')
        col_uri_a = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a')

        self.assertEqual(1, len(self.graph.tail.read))
        self.assertEqual(col_uri_a, self.graph.tail.read[0].uri)
        self.assertListEqual(['a'], node_visitor.var_columns.get('X'))
        self.assertIn('X', node_visitor.files.keys())

    def test_when_cross_val_score_then_save_target_and_features(self):
        value = "from sklearn.model_selection import cross_val_score\n" \
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0:3]\n" \
                "y = df.iloc[:, 3]\n" \
                "cross_val_score(estimator=None, X=X, y=y, cv=10)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])
        col_uri_a = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a')
        col_uri_b = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')
        col_uri_c = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'c')
        col_uri_d = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'd')

        self.assertEqual(1, len(self.graph.tail.targets))
        self.assertListEqual([col_uri_d], [x.uri for x in self.graph.tail.targets])
        self.assertEqual(3, len(self.graph.tail.features))
        self.assertListEqual([col_uri_a, col_uri_b, col_uri_c], [x.uri for x in self.graph.tail.features])

    def test_when_cross_val_score_without_keywords_then_save_target_and_features(self):
        value = "from sklearn.model_selection import cross_val_score\n" \
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0:3]\n" \
                "y = df.iloc[:, 3]\n" \
                "cross_val_score(None, X, y, cv=10)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])
        col_uri_a = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a')
        col_uri_b = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')
        col_uri_c = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'c')
        col_uri_d = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'd')

        self.assertEqual(1, len(self.graph.tail.targets))
        self.assertListEqual([col_uri_d], [x.uri for x in self.graph.tail.targets])
        self.assertEqual(3, len(self.graph.tail.features))
        self.assertListEqual([col_uri_a, col_uri_b, col_uri_c], [x.uri for x in self.graph.tail.features])

    def test_when_train_test_split_then_save_target_and_features(self):
        value = "from sklearn.model_selection import train_test_split\n" \
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0:3]\n" \
                "y = df.iloc[:, 3]\n" \
                "X_train, X_test, y_train, y_test = train_test_split(X, y)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])
        col_uri_a = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a')
        col_uri_b = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'b')
        col_uri_c = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'c')
        col_uri_d = util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'd')

        self.assertEqual(1, len(self.graph.tail.targets))
        self.assertListEqual([col_uri_d], [x.uri for x in self.graph.tail.targets])
        self.assertEqual(3, len(self.graph.tail.features))
        self.assertListEqual([col_uri_a, col_uri_b, col_uri_c], [x.uri for x in self.graph.tail.features])

    def test_when_dropping_column_then_save_all_columns_except_one_drop(self):
        value = "import pandas as pd\n" \
                "X = df.drop(['Price'], axis=1)\n" \
                "y = df['Price']"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['Price', 'a', 'b'], 'df')

        self.assertEqual(2, len(node_visitor.var_columns.get('X')))
        self.assertListEqual(['a', 'b'], node_visitor.var_columns.get('X'))
        self.assertEqual(1, len(node_visitor.var_columns.get('y')))
        self.assertListEqual(['Price'], node_visitor.var_columns.get('y'))

    def test_when_cross_val_score_then_save_target_and_features_from_class_object(self):
        value = "cvs_scores = cross_val_score(knn_normal, x_pca_test, y_pca_test, cv=5)"
        pass

    def test_when_cross_val_score_then_save_target_and_features_from_saved_variable(self):
        pass


if __name__ == '__main__':
    unittest.main()
