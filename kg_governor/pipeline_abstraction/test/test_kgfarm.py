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


def create_column_uri(file: str, column: str):
    return util.create_column_name(SOURCE, DATASET_NAME, file, column)


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
    @unittest.skip
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

    @unittest.skip
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

    def test_when_dataframe_is_split_twice_it_reads_the_correct_column(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n"\
                "y = df[['a']][:]"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b'])
        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a'),
            self.graph.tail.read[0].uri)

    def test_when_dataframe_is_slice_with_attribute_link_correct_columns(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n"\
                "y = df[df.a, :]"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b'])

        # self.assertEqual(len(self.graph.tail.read[0].uri), 1)
        self.assertEqual(
            util.create_column_name(SOURCE, DATASET_NAME, 'file.csv', 'a'),
            self.graph.tail.read[0].uri)

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

    def test_parsing_column_does_not_affect_column_name(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')\n" \
                "col = 'Small Bags'\n" \
                "df[col] = 1"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['Small Bags'])

        self.assertIn('Small Bags', node_visitor.var_columns.get('df'))

    def test_when_variable_assigned_column_then_column_is_read_when_passed_as_parameter(self):
        value = "import pandas as pd\n\n" \
                "df = pd.read_csv('file.csv')\n" \
                "col = 'b'\n" \
                "df.drop(col, 1, inplace=True)\n"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['b'])

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

    def test_when_separating_dataframe_then_correct_column_in_slice_are_saved_for_the_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.loc[:, 'a':'b']"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'], 'df')
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')

        self.assertEqual(2, len(self.graph.tail.read))
        self.assertListEqual([col_uri_a, col_uri_b], [x.uri for x in self.graph.tail.read])
        # self.assertEqual([col_uri_a, col_uri_b], [x.uri for x in self.graph.tail.read])
        # self.assertEqual(, self.graph.tail.read[1].uri)
        self.assertListEqual(['a', 'b'], node_visitor.var_columns.get('X'))

    def test_when_dropping_column_in_place_it_changes_the_original_variable(self):
        value = "from sklearn.model_selection import train_test_split\n"\
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "y = df[['type']][:]\n" \
                "x = df.drop(['type'], axis=1, inplace=True)\n" \
                "x = df.iloc[:, :]\n" \
                "x_train, x_test, y_train, y_test = train_test_split(x, y)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['type', 'a', 'b'], 'df')
        col_uri_type = create_column_uri('file.csv', 'type')
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')

        self.assertEqual(1, len(self.graph.tail.targets))
        self.assertListEqual([col_uri_type], [x.uri for x in self.graph.tail.targets])
        self.assertEqual(2, len(self.graph.tail.features))
        self.assertListEqual([col_uri_a, col_uri_b], [x.uri for x in self.graph.tail.features])

    def test_when_dropping_column_with_multiple_word_in_place_it_changes_the_original_variable(self):
        value = "from sklearn.model_selection import train_test_split\n"\
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "y = df[['a']][:]\n" \
                "x = df.drop(['a'], axis=1, inplace=True)\n" \
                "x = df.iloc[:, :]\n" \
                "x.drop(['Small Bags'], axis=1, inplace=True)\n" \
                "x_train, x_test, y_train, y_test = train_test_split(x, y)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['Small Bags', 'a', 'b'], 'df')
        col_uri_type = create_column_uri('file.csv', 'Small Bags')
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')

        self.assertEqual(1, len(self.graph.tail.targets))
        self.assertListEqual([col_uri_a], [x.uri for x in self.graph.tail.targets])
        self.assertEqual(1, len(self.graph.tail.features))
        self.assertListEqual([col_uri_b], [x.uri for x in self.graph.tail.features])
        self.assertEqual(1, len(self.graph.tail.not_features))
        self.assertListEqual([col_uri_type], [x.uri for x in self.graph.tail.not_features])

    def test_weird_read_thing(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "x = df.iloc[:, :]\n" \

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b'], 'df')

        self.assertEqual(2, len(node_visitor.graph_info.tail.read))

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

    def test_when_passing_a_column_to_a_data_frame_it_saves_to_column_to_assign_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df['a']"
        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])

        col_uri_a = create_column_uri('file.csv', 'a')

        self.assertEqual(1, len(self.graph.tail.read))
        self.assertListEqual([col_uri_a], [x.uri for x in self.graph.tail.read])
        self.assertListEqual(['a'], node_visitor.var_columns.get('X'))

    def test_when_passing_a_column_to_a_data_frame_inside_double_array_it_saves_to_column_to_assign_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df[['a']][:]"
        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])

        col_uri_a = create_column_uri('file.csv', 'a')

        self.assertEqual(1, len(self.graph.tail.read))
        self.assertListEqual([col_uri_a], [x.uri for x in self.graph.tail.read])
        self.assertListEqual(['a'], node_visitor.var_columns.get('X'))

    def test_when_passing_a_column_to_loc_it_saves_to_column_to_assign_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.loc[:, 'a']"
        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])

        col_uri_a = create_column_uri('file.csv', 'a')

        self.assertEqual(1, len(self.graph.tail.read))
        self.assertListEqual([col_uri_a], [x.uri for x in self.graph.tail.read])
        self.assertListEqual(['a'], node_visitor.var_columns.get('X'))

    def test_when_passing_a_list_of_column_to_loc_it_saves_to_columns_to_assign_variable(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.loc[:, ['a', 'b', 'c']]"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])

        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')
        col_uri_c = create_column_uri('file.csv', 'c')

        self.assertEqual(3, len(self.graph.tail.read))
        self.assertListEqual([col_uri_a, col_uri_b, col_uri_c], [x.uri for x in self.graph.tail.read])
        self.assertListEqual(['a', 'b', 'c'], node_visitor.var_columns.get('X'))

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
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')
        col_uri_c = create_column_uri('file.csv', 'c')
        col_uri_d = create_column_uri('file.csv', 'd')

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

    def test_when_dropping_column_it_appears_as_not_selected_features(self):
        value = "from sklearn.model_selection import train_test_split\n" \
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.iloc[:, 0:3]\n" \
                "y = df.iloc[:, 3]\n" \
                "X.drop('b', 1, inplace=True)\n" \
                "X_train, X_test, y_train, y_test = train_test_split(X, y)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')
        col_uri_c = create_column_uri('file.csv', 'c')
        col_uri_d = create_column_uri('file.csv', 'd')

        self.assertNotIn(self.graph.tail.targets[0], self.graph.tail.features)
        self.assertNotIn(col_uri_b, self.graph.tail.features)
        self.assertIn(col_uri_b, [x.uri for x in self.graph.tail.not_features])
        self.assertEqual(1, len(self.graph.tail.not_features))

    def test_when_dropping_column_it_appears_as_not_selected_features_using_drop(self):
        value = "from sklearn.model_selection import train_test_split\n" \
                "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.loc[:, ['a', 'b', 'c']]\n" \
                "y = df['d']\n" \
                "X.drop('a', 1, inplace=True)\n" \
                "X_train, X_test, y_train, y_test = train_test_split(X, y)"

        parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a', 'b', 'c', 'd'])
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')
        col_uri_c = create_column_uri('file.csv', 'c')
        col_uri_d = create_column_uri('file.csv', 'd')

        self.assertEqual(2, len(self.graph.tail.features))
        self.assertNotIn(self.graph.tail.targets[0], self.graph.tail.features)
        self.assertNotIn(col_uri_a, self.graph.tail.features)
        self.assertIn(col_uri_a, [x.uri for x in self.graph.tail.not_features])
        self.assertEqual(1, len(self.graph.tail.not_features))

    def test_when_cross_val_score_then_save_target_and_features_from_class_object(self):
        value = "cvs_scores = cross_val_score(knn_normal, x_pca_test, y_pca_test, cv=5)"
        pass

    def test_when_cross_val_score_then_save_target_and_features_from_saved_variable(self):
        pass

    def test_when_separating_columns_by_column_names_then_columns_are_correctly_saved(self):
        value = "df.iloc[:, 'a':'c']"
        pass

    def test_when_dropping_a_column_without_inplace_it_do_not_modify_the_dataframe(self):
        value = "x = df.drop('a', axis=1, inplace=False)"
        columns = ['a', 'b', 'c', 'd']

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', columns)
        col_uri_a = create_column_uri('file.csv', 'a')
        col_uri_b = create_column_uri('file.csv', 'b')
        col_uri_c = create_column_uri('file.csv', 'c')
        col_uri_d = create_column_uri('file.csv', 'd')

        self.assertListEqual(['b', 'c', 'd'], node_visitor.var_columns.get('x'))
        self.assertListEqual(columns, node_visitor.var_columns.get('df'))

    def test_specific_pipeline_workflow(self):
        columns = ['Small Bags', 'Large Bags', 'type', 'd']
        value = "from sklearn.model_selection import train_test_split\n" \
                "from sklearn.preprocessing import StandardScaler\n" \
                "y = df[['type']][:]\n" \
                "x = df.drop(['type'], axis=1, inplace=True)\n" \
                "x = df.iloc[:, :]\n"\
                "x.drop(['Small Bags'], axis=1)\n"\
                "sc = StandardScaler()\n" \
                "x = sc.fit_transform(x)\n" \
                "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', columns)

        col_uri_small_bags = create_column_uri('file.csv', 'Small Bags')
        col_uri_large_bags = create_column_uri('file.csv', 'Large Bags')
        col_uri_type = create_column_uri('file.csv', 'type')
        col_uri_d = create_column_uri('file.csv', 'd')

        self.assertListEqual(['type'], node_visitor.var_columns.get('y'))
        self.assertListEqual(['Large Bags', 'd'], node_visitor.var_columns.get('x'))
        self.assertListEqual([col_uri_large_bags,col_uri_d], [x.uri for x in self.graph.tail.features])
        self.assertListEqual([col_uri_type], [x.uri for x in self.graph.tail.targets])
        self.assertListEqual([col_uri_small_bags], [x.uri for x in self.graph.tail.not_features])
        # self.assertListEqual(columns, node_visitor.var_columns.get('df'))

    def test_empty_value_in_tuple_return_all_element(self):
        value = "import pandas as pd\n" \
                "df = pd.read_csv('file.csv')\n" \
                "X = df.loc[:,]"

        node_visitor = parse_and_visit_node_with_file(value, self.graph, 'file.csv', ['a'])
        col_uri_a = create_column_uri('file.csv', 'a')

        self.assertEqual(
            node_visitor.var_columns['X'],
            ['a']
        )

if __name__ == '__main__':
    unittest.main()
