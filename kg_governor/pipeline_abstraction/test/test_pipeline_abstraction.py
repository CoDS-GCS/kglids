import unittest
import ast
import pandas as pd

import src.Calls as Calls
import src.util as util
import src.datatypes as DataTypes
from src.datatypes import GraphInformation
from src.Calls import pd_dataframe, File, packages
from src.pipeline_abstraction import NodeVisitor, insert_parameter
from src.util import ControlFlow

kglids_library = "http://kglids.org/pipeline/library/"
FILENAME = "test.py"
SOURCE = "<SOURCE>"
DATASET_NAME = "<DATASET_NAME>"


def parse_and_visit_node(lines: str, graph: GraphInformation) -> NodeVisitor:
    parse_tree = ast.parse(lines)
    node_visitor = NodeVisitor(graph_information=graph)
    node_visitor.visit(parse_tree)
    return node_visitor


def generate_id(node: DataTypes.Node):
    i = 1
    while node is not None:
        node.generate_uri(SOURCE, DATASET_NAME, FILENAME, i)
        node = node.next
        i += 1


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphInformation(
            python_file_name=FILENAME,
            source=SOURCE,
            dataset_name=DATASET_NAME
        )


class VisitNode(Test):
    def test_visit_node_create_node(self):
        value = "a = 'element'"

        parse_and_visit_node(value, self.graph)

        generate_id(self.graph.head)
        self.assertEqual(self.graph.head.text, value)
        self.assertEqual(self.graph.head,
                         self.graph.tail)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertIsNone(self.graph.head.next)


class VisitNameNode(Test):
    def test_ast_name_node_return_string(self):
        value = 'my_awesome_name'
        name = ast.Name(value)

        result = NodeVisitor().visit_Name(name)
        self.assertEqual(result, value)


class VisitConstantNode(Test):
    def test_ast_constant_node_return_string(self):
        value = 'my_awesome_name'
        cst = ast.Constant(value)

        result = NodeVisitor().visit_Constant(cst)
        self.assertEqual(result, value)


class VisitArgNode(Test):
    def test_ast_arg_node_return_arg_name_as_string(self):
        value = "arg1"
        arg = ast.arg(value)

        result = NodeVisitor().visit_arg(arg)
        self.assertEqual(result, value)


class VisitAliasNode(Test):
    def test_ast_alias_return_tuple(self):
        alias = ast.alias(name='pandas', asname='pd')

        name, as_name = NodeVisitor().visit_alias(alias)
        self.assertEqual(name, 'pandas')
        self.assertEqual(as_name, 'pd')


class VisitKeywordNode(Test):
    def test_ast_keyword_return_tuple(self):
        arg = 'key'
        value = 'value'
        keyword = ast.keyword(arg=arg, value=ast.Constant(value))

        k, v = NodeVisitor().visit_keyword(keyword)
        self.assertEqual(k, arg)
        self.assertEqual(v, value)

    @unittest.skip("not working")
    def test_ast_keyword_to_continue(self):
        pass
        self.assertEqual(True, False)


class VisitImportNode(Test):
    def test_ast_import_node_create_node(self):
        value = "import pandas as pd"
        parse_and_visit_node(value, self.graph)

        generate_id(self.graph.head)
        lib_uri = util.create_import_uri('pandas')
        self.assertEqual(self.graph.head.text, value)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(lib_uri,
                         self.graph.head.calls[0].uri)
        self.assertIn(lib_uri, self.graph.libraries.keys())
        self.assertEqual(lib_uri,
                         self.graph.libraries.get(lib_uri).uri)
        self.assertEqual(1, len(self.graph.head.control_flow))
        self.assertEqual(util.ControlFlow.IMPORT.value,
                         self.graph.head.control_flow.pop())

    def test_ast_import_node_with_multi_import_create_multi_edge_and_node(self):
        value = "import pandas, sklearn"
        parse_and_visit_node(value, self.graph)

        lib_uri_1 = util.create_import_uri('pandas')
        lib_uri_2 = util.create_import_uri('sklearn')

        self.assertEqual(2, len(self.graph.head.calls))
        self.assertEqual(lib_uri_1, self.graph.head.calls[0].uri)
        self.assertEqual(lib_uri_2, self.graph.head.calls[1].uri)
        self.assertIn(lib_uri_1, self.graph.libraries.keys())
        self.assertIn(lib_uri_2, self.graph.libraries.keys())
        self.assertEqual(lib_uri_1, self.graph.libraries.get(lib_uri_1).uri)
        self.assertEqual(lib_uri_2, self.graph.libraries.get(lib_uri_2).uri)
        self.assertEqual(1, len(self.graph.head.control_flow))
        self.assertEqual(util.ControlFlow.IMPORT.value,
                         self.graph.head.control_flow.pop())


class VisitImportFromNode(Test):
    def test_ast_import_from_node_create_node(self):
        value = "from sklearn import preprocessing"
        parse_and_visit_node(value, self.graph)

        lib_uri_1 = util.create_import_uri('sklearn')
        lib_uri_2 = util.create_import_uri('sklearn.preprocessing')

        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(lib_uri_2, self.graph.head.calls[0].uri)
        self.assertIn(lib_uri_1, self.graph.libraries.keys())
        self.assertEqual(lib_uri_1, self.graph.libraries.get(lib_uri_1).uri)
        self.assertIn(lib_uri_2, self.graph.libraries.get(lib_uri_1).contain.keys())
        self.assertEqual(lib_uri_2, self.graph.libraries.get(lib_uri_1).contain.get(lib_uri_2).uri)

    def test_ast_import_from_node_with_multi_import_create_multi_edge_and_node(self):
        value = "from sklearn import preprocessing, svm"
        parse_and_visit_node(value, self.graph)

        lib_uri_1 = util.create_import_uri('sklearn')
        lib_uri_2 = util.create_import_uri('sklearn.preprocessing')
        lib_uri_3 = util.create_import_uri('sklearn.svm')

        self.assertEqual(2, len(self.graph.head.calls))
        self.assertEqual(lib_uri_2, self.graph.head.calls[0].uri)
        self.assertEqual(lib_uri_3, self.graph.head.calls[1].uri)
        self.assertIn(lib_uri_1, self.graph.libraries.keys())
        self.assertEqual(lib_uri_1, self.graph.libraries.get(lib_uri_1).uri)
        self.assertIn(lib_uri_2, self.graph.libraries.get(lib_uri_1).contain.keys())
        self.assertIn(lib_uri_3, self.graph.libraries.get(lib_uri_1).contain.keys())
        self.assertEqual(lib_uri_2, self.graph.libraries.get(lib_uri_1).contain.get(lib_uri_2).uri)
        self.assertEqual(lib_uri_3, self.graph.libraries.get(lib_uri_1).contain.get(lib_uri_3).uri)

    def test_import_star_is_not_selected_as_followed(self):
        value = 'from sklearn import *'
        parse_and_visit_node(value, self.graph)

        lib_uri_1 = util.create_import_uri('sklearn')
        lib_uri_2 = util.create_import_uri('sklearn.*')

        self.assertEqual(0, len(self.graph.head.calls))
        self.assertIn(lib_uri_1, self.graph.libraries.keys())
        self.assertEqual(lib_uri_1, self.graph.libraries.get(lib_uri_1).uri)
        self.assertNotIn(lib_uri_2, self.graph.libraries.get(lib_uri_1).contain.keys())


class VisitAssignNode(Test):
    def test_ast_assign_create_node(self):
        value = "train_path = '/kaggle/input/titanic/train.csv'"
        parse_and_visit_node(value, self.graph)

        generate_id(self.graph.head)

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(value, self.graph.head.text)

    @unittest.skip
    def test_ast_assign_if_file_save_file_and_link_file(self):
        value = "train_path = '/kaggle/input/titanic/train.csv'"
        parse_and_visit_node(value, self.graph)

        file_id = util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv')

        # self.assertIn('train.csv', result.keys())
        # self.assertIsNotNone(result.get('train.csv'))
        # self.assertEqual(result.get('train.csv').uri, file_id)
        # self.assertIn(1, result.keys())
        # self.assertIn('1-train.csv', result.keys())
        # self.assertIsNotNone(result.get('1-train.csv'))

    def test_ast_assign_if_file_associate_file_to_variable(self):
        value = "train_path = '/kaggle/input/titanic/train.csv'"
        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertIn('train_path', node_visitor.files.keys())
        self.assertIsNotNone(node_visitor.files.get('train_path'))
        self.assertEqual(node_visitor.files.get('train_path').filename, 'train.csv')

    def test_ast_assign_save_file_and_dataframe_to_variable_and_connect_column(self):
        value = "y = train['Survived']"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived'])
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)
        generate_id(self.graph.head)
        column_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')

        self.assertIn(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                      self.graph.tail.uri)
        self.assertEqual(1, len(self.graph.tail.read))
        self.assertEqual(column_uri, self.graph.tail.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(column_uri, self.graph.files.get('train.csv').contain.pop().uri)
        self.assertIn('y', node_visitor.files.keys())
        self.assertEqual(node_visitor.files.get('y').filename, 'train.csv')
        self.assertIn('y', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('y'), pd_dataframe)

    def test_ast_assign_save_column_when_index_is_used(self):
        value = "y = train[0]"
        tree = ast.parse(value)

        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived'])
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)
        column_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')
        generate_id(self.graph.head)

        self.assertIn(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                      self.graph.tail.uri)
        self.assertEqual(1, len(self.graph.tail.read))
        self.assertEqual(column_uri, self.graph.tail.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(column_uri, self.graph.files.get('train.csv').contain.pop().uri)

    # TODO: VERIFY THIS TEST
    def test_ast_assign_separate_nested_class_into_multiple_nodes(self):
        value = "print(i)\n" \
                "X = pd.get_dummies(train.drop('Survived', axis=1))"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.variables['train'] = pd_dataframe

        node_visitor.visit(tree)
        generate_id(self.graph.head)

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual("print(i)", self.graph.head.text)
        self.assertEqual("train.drop('Survived', axis=1)", self.graph.head.next.text)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual("X = pd.get_dummies(train.drop('Survived', axis=1))",
                         self.graph.head.next.next.text)
        self.assertEqual('labels', self.graph.head.next.parameters[0].parameter)
        self.assertEqual('axis', self.graph.head.next.parameters[1].parameter)
        self.assertEqual('Survived', self.graph.head.next.parameters[0].parameter_value)
        self.assertEqual('1', self.graph.head.next.parameters[1].parameter_value)
        self.assertEqual(7, len(self.graph.head.next.parameters))
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 3),
                         self.graph.head.next.next.uri)
        self.assertEqual(8, len(self.graph.head.next.next.parameters))
        self.assertIsNone(self.graph.head.next.next.next)

    def test_ast_assign_transfer_return_type_to_all_tuple_variables(self):
        value = "from sklearn.model_selection import train_test_split\n" \
                "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)"
        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertIn('X_train', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('X_train'), pd_dataframe)
        self.assertIn('X_val', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('X_val'), pd_dataframe)
        self.assertIn('y_train', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('y_train'), pd_dataframe)
        self.assertIn('y_val', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('y_val'), pd_dataframe)

    def test_ast_assign_transfer_file_from_array_to_variable(self):
        value = "full = pd.concat([train,test], ignore_index=True)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.alias['pd'] = 'pandas'

        node_visitor.visit(tree)

        self.assertIn("full", node_visitor.files)

    def test_ast_assign_bin_op_return_types_to_variable(self):
        value = "a = 'A'\n" \
                "p = (train.isna().sum()/len(train) * 100).sort_values(ascending=False)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables['train'] = pd_dataframe
        node_visitor.visit(tree)

        lib_uri = util.create_import_uri('pandas.DataFrame.sort_values')
        self.assertEqual(0, len(self.graph.head.next.parameters))
        self.assertEqual(5, len(self.graph.head.next.next.parameters))
        self.assertEqual(0, len(self.graph.head.next.next.next.parameters))
        self.assertEqual(8, len(self.graph.tail.parameters))
        param = [x for x in self.graph.tail.parameters if x.parameter == 'ascending'][0]
        self.assertEqual('False', param.parameter_value)
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)

    def test_ast_assign_bin_op_when_subscript_left_branch_return_types_to_variable(self):
        value = "a = 'A'\n" \
                "p = (train['a'] - 100).sort_values(ascending=False)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables['train'] = pd_dataframe
        node_visitor.visit(tree)

        lib_uri = util.create_import_uri('pandas.DataFrame.sort_values')
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)

    @unittest.skip("not working")
    def test_ast_assign_multiple_right_hand_value_to_multiple_left_hand_value(self):
        value = "X_train, X_test = X[train_index], X[test_index]"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)

        node_visitor.visit(tree)

        self.assertTrue(False)

    @unittest.skip("not working")
    def test_ast_assign_format_list_comp_correctly(self):
        value = "cols_with_missing = [col for col in X_train.columns " \
                "if X_train[col].isnull().any()]"

        self.assertTrue(False)

    def test_ast_assign_format_library_SVC_correctly(self):
        value = "from sklearn.svm import SVC\n" \
                "model5 = SVC()"
        parse_and_visit_node(value, self.graph)

        lib_uri = util.create_import_uri('sklearn.svm.SVC')
        sklearn = util.create_import_uri('sklearn')
        svm = util.create_import_uri('sklearn.svm')

        self.assertEqual(1, len(self.graph.tail.calls))
        self.assertEqual(lib_uri, self.graph.tail.calls[0].uri)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertIsNotNone(self.graph.libraries.get(sklearn))
        self.assertEqual(1, len(self.graph.libraries.get(sklearn).contain))
        self.assertIsNotNone(self.graph.libraries.get(sklearn).contain.get(svm))
        self.assertEqual(1, len(self.graph.libraries.get(sklearn).contain.get(svm).contain))
        self.assertIsNotNone(self.graph.libraries.get(sklearn).contain.get(svm).contain.get(lib_uri))

    @unittest.skip("not working")
    def test_ast_assign_associate_list_to_variable(self):
        value = "arr = ['a', 'b']"
        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertEqual(node_visitor.variables['arr'], ['a', 'b'])

    def test_ast_assign_with_more_complex_version(self):
        value = "player = pandas.read_csv('../input/predict-nhl-player-salaries/train.csv', encoding='ISO-8859-1').fillna(0)"
        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertIn('player', node_visitor.files.keys())
        self.assertEqual('train.csv', node_visitor.files.get('player').filename)


class VisitExprNode(Test):
    @unittest.skip("not working")
    def test_expr_attribute(self):
        value = "gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold,\n" \
                " scoring='accuracy', n_jobs= 4, verbose = 1)\n" \
                "gsSVMC.fit(X_train,Y_train)"
        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertTrue(False)

    def test_expr_node_value_remove_comment_line(self):
        value = '""" # **Variables associated with SalePrice** """'
        parse_and_visit_node(value, self.graph)
        self.assertIsNone(self.graph.head)

    def test_expr_when_call_method_give_it_to_visit_call(self):
        value = 'from sklearn.model_selection import cross_val_score\n' \
                'print("Lasso Cross Validation: ", cross_val_score(lasso, X, y, cv=5))'
        parse_and_visit_node(value, self.graph)

        fct = util.create_import_uri('sklearn.model_selection.cross_val_score')

        self.assertEqual(1, len(self.graph.tail.previous.calls))
        self.assertEqual(fct, self.graph.tail.previous.calls[0].uri)


class VisitAttributeNode(Test):
    def test_ast_attribute_verify_if_element_is_a_column(self):
        value = "value = all_df.PoolQC.sum()"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'all_df': File("train.csv")}
        node_visitor.variables = {'all_df': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['PoolQC'])
        node_visitor.visit(tree)

        generate_id(self.graph.head)
        col1 = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'PoolQC')
        lib_uri = util.create_import_uri('pandas.DataFrame.sum')

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(1, len(self.graph.head.read))
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(lib_uri, self.graph.head.calls[0].uri)
        self.assertEqual(col1, self.graph.head.read.pop().uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(col1, self.graph.files.get('train.csv').contain.pop().uri)

    def test_ast_attribute_view_column_as_dataframe(self):
        value = "value = all_df.PoolQC.sum()"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'all_df': File("train.csv")}
        node_visitor.variables = {'all_df': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['PoolQC'])
        node_visitor.visit(tree)

        self.assertIn('value', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get("value").name,
                         pd_dataframe.name)
        self.assertEqual(len(node_visitor.columns), 0)

    def test_attribute_return_path_to_package_and_base_package(self):
        node1 = ast.parse('train.sum()').body[0].value.func
        node2 = ast.parse('train.isnull()').body[0].value.func
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'train': pd_dataframe}
        path1, base_package1, file = node_visitor.visit_Attribute(node1)
        path2, base_package2, file = node_visitor.visit_Attribute(node2)

        self.assertEqual('pandas.DataFrame.sum', path1)
        self.assertEqual('train', base_package1)
        self.assertEqual('pandas.DataFrame.isnull', path2)
        self.assertEqual('train', base_package2)

    def test_attribute_return_path_to_package_and_base_package_of_attributes(self):
        node1 = ast.parse('train.isnull().sum()').body[0].value.func
        node2 = ast.parse('train.sum().isnull()').body[0].value.func
        node_visitor = NodeVisitor(self.graph)
        node_visitor.graph_info.add_node('')
        node_visitor.variables = {'train': pd_dataframe}
        path1, base_package1, _ = node_visitor.visit_Attribute(node1)
        path2, base_package2, _ = node_visitor.visit_Attribute(node2)

        self.assertEqual('pandas.DataFrame.sum', path1)
        self.assertEqual('train', base_package1)
        self.assertEqual('pandas.DataFrame.isnull', path2)
        self.assertEqual('train', base_package2)

    def test_attribute_when_column_present_return_dataframe(self):
        value = 'train = pandas.read_csv("train.csv")\n' \
                'x = train.sum().Age.sort_values()'
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])

        node_visitor.visit(tree)
        self.assertEqual(pd_dataframe.name, node_visitor.variables.get('x').name)
        self.assertEqual(1, len(self.graph.tail.read))

    def test_attribute_when_column_is_present_add_read_column(self):
        value = 'train = pandas.read_csv("train.csv")\n' \
                'x = train.Age'
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])

        node_visitor.visit(tree)

        self.assertEqual(1, len(self.graph.tail.read))
        self.assertEqual(util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age'),
                         self.graph.tail.read[0].uri)


class VisitCallNode(Test):
    def test_call_argument_subscript_are_return_as_type(self):
        value = "pd.DataFrame(df[col], 25)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'df': pd_dataframe}
        node_visitor.alias = {'pd': 'pandas'}

        node_visitor.visit(tree)

        generate_id(self.graph.head)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(5, len(self.graph.head.parameters))
        self.assertEqual('data', self.graph.head.parameters[0].parameter)
        self.assertEqual('DataFrame', self.graph.head.parameters[0].parameter_value)
        self.assertEqual('index', self.graph.head.parameters[1].parameter)
        self.assertEqual('25', self.graph.head.parameters[1].parameter_value)
        self.assertEqual(self.graph.head, self.graph.tail)

    def test_ast_calls_remove_unused_columns(self):
        value = 'sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)\n' \
                'a = "A"'
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.working_file = pd.DataFrame(columns=['Embarked', 'Survived'])
        node_visitor.visit(tree)

        self.assertNotIn('Embarked', node_visitor.columns)
        self.assertNotIn('Survived', node_visitor.columns)
        self.assertEqual(len(node_visitor.columns), 0)

    @unittest.skip("not working")
    def test_ast_call_format_attribute_argument_correctly(self):
        value = "train_test_dtype_info=pd.DataFrame(train_test.dtypes,columns=['DataTypes'])"

        self.assertTrue(False)

    def test_ast_call_separate_consecutive_call_into_multiple_nodes(self):
        value = "value = train.isna().sum()"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)

        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')
        isna_uri = util.create_import_uri("pandas.DataFrame.isna")
        sum_uri = util.create_import_uri("pandas.DataFrame.sum")

        self.assertIsNotNone(self.graph.head)
        self.assertIsNotNone(self.graph.head.next)
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(1, len(self.graph.head.next.calls))
        self.assertEqual(isna_uri, self.graph.head.calls[0].uri)
        self.assertEqual(sum_uri, self.graph.head.next.calls[0].uri)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertEqual(1, len(self.graph.libraries.get(pandas_uri).contain))
        self.assertEqual(2, len(self.graph.libraries.get(pandas_uri).contain.get(dataframe_uri).contain))

    def test_ast_call_separate_consecutive_call_into_multiple_nodes_more_complex(self):
        value = "data['Age'] = data.groupby(['Pclass','Sex']).transform(lambda x: x.fillna(x.median()))"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'data': pd_dataframe}

        node_visitor.visit(tree)

        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')
        groupby_uri = util.create_import_uri("pandas.DataFrame.groupby")
        transform_uri = util.create_import_uri("pandas.DataFrame.transform")

        self.assertIsNotNone(self.graph.head)
        self.assertIsNotNone(self.graph.head.next)
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(9, len(self.graph.head.parameters))
        self.assertEqual(1, len(self.graph.head.next.calls))
        self.assertEqual(2, len(self.graph.head.next.parameters))
        self.assertEqual(groupby_uri, self.graph.head.calls[0].uri)
        self.assertEqual(transform_uri, self.graph.head.next.calls[0].uri)
        self.assertEqual("['Pclass', 'Sex']", self.graph.head.parameters[0].parameter_value)
        self.assertEqual("by", self.graph.head.parameters[0].parameter)
        self.assertEqual("lambda x: x.fillna(x.median())", self.graph.head.next.parameters[0].parameter_value)
        self.assertEqual("func", self.graph.head.next.parameters[0].parameter)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertEqual(1, len(self.graph.libraries.get(pandas_uri).contain))
        self.assertEqual(2, len(self.graph.libraries.get(pandas_uri).contain.get(dataframe_uri).contain))

    # def test_column_call_in_calls_then_subscript(self):
    #     value = "df = pandas.read_csv('train.csv')\n" \
    #             "x = df.groupby('X')['Y']"
    #     tree = ast.parse(value)
    #     node_visitor = NodeVisitor(self.graph)
    #     node_visitor.working_file = pd.DataFrame(columns=['X', 'Y'])
    #
    #     node_visitor.visit(tree)
    #
    #     self.assertEqual(2, len(self.graph.tail.read))


class VisitIfNode(Test):
    def test_ast_if_node_link_to_loop_control_flow(self):
        value = 'if True: \n\tprint(True)'
        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(1, len(self.graph.head.control_flow))
        self.assertEqual(ControlFlow.CONDITIONAL.value, self.graph.head.control_flow.pop())
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(1, len(self.graph.head.next.control_flow))
        self.assertEqual(ControlFlow.CONDITIONAL.value, self.graph.head.next.control_flow.pop())

    def test_ast_if_node_link_or_else_to_loop_control_flow(self):
        # TODO: FIND A WAY TO DISTINGUISH BETWEEN FOR AND ELSE
        value = "if True: \n\ta = 'b' \nelse:\n\ta = 'c'"
        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(1, len(self.graph.head.control_flow))
        self.assertEqual(ControlFlow.CONDITIONAL.value, self.graph.head.control_flow.pop())
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(1, len(self.graph.head.next.control_flow))
        self.assertEqual(ControlFlow.CONDITIONAL.value, self.graph.head.next.control_flow.pop())
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 3),
                         self.graph.head.next.next.uri)
        self.assertEqual(1, len(self.graph.head.next.next.control_flow))
        self.assertEqual(ControlFlow.CONDITIONAL.value, self.graph.head.next.next.control_flow.pop())


class VisitDictNode(Test):
    @unittest.skip("not working")
    def test_dict_node_subscript_value_return_type(self):
        value = "{'PassengerId': test_df['PassengerId'], 'Survived': predictions}"

        self.assertTrue(False)


class VisitListNode(Test):
    def test_ast_list_return_array(self):
        value = "['a', 'b', 'c', 'd']"
        result = NodeVisitor().visit_List(ast.parse(value).body[0].value)
        self.assertEqual(len(result), 4)
        self.assertEqual('a', result[0])
        self.assertEqual('b', result[1])
        self.assertEqual('c', result[2])
        self.assertEqual('d', result[3])

    @unittest.skip("not working")
    def test_ast_list_continue(self):
        pass
        self.assertEqual(True, False)


class VisitForNode(Test):
    def test_ast_for_node_link_to_control_flow(self):
        value = "for i in range(10):\n\tprint(i)"
        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(1, len(self.graph.head.control_flow))
        self.assertEqual(ControlFlow.LOOP.value, self.graph.head.control_flow.pop())
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(1, len(self.graph.head.next.control_flow))
        self.assertEqual(ControlFlow.LOOP.value, self.graph.head.next.control_flow.pop())

    def test_ast_for_node_pass_iter_variable_to_every_loop_variable(self):
        value = "for model in models:\n" + \
                "\ty = pd.read_csv(model)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'models': ['file1.csv', 'file2.csv']}
        node_visitor.alias = {'pd': 'pandas'}
        self.graph.files['file1.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['file2.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor.visit(tree)

        generate_id(self.graph.head)
        file1_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file1.csv')
        file2_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file2.csv')

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(2, len(self.graph.head.next.read))
        self.assertEqual(file1_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(file2_uri, self.graph.head.next.read[1].uri)

    def test_ast_for_node_loop_variable_is_deleted_after_loop(self):
        value = "for model in models:\n" + \
                "\tprint(\"Model \", i,\":\", model)\n" + \
                "\tprint(\"ACC: \", fitAndPredict(model))\n"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'models': ['file1.csv', 'file2.csv']}

        node_visitor.visit(tree)
        self.assertNotIn('model', node_visitor.variables.keys())

    # TODO: VERIFY THIS TEST
    def test_ast_for_node_variables_within_the_loop_are_delete_when_exiting_loop(self):
        value = "for model in models:\n" + \
                "\ty = pd.read_csv(model)"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {'models': ['file1.csv', 'file2.csv']}

        node_visitor.visit(tree)
        self.assertNotIn('y', node_visitor.variables.keys())

    def test_ast_for_node_return_column_visited_in_index(self):
        value = "for feature in ['Embarked', 'Survived']:\n" \
                "\ttrain[feature] = 1"

        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Embarked', 'Survived'])
        node_visitor.visit(tree)

        generate_id(self.graph.head)
        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Embarked')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(2, len(self.graph.head.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.read[1].uri)
        self.assertEqual(2, len(self.graph.files.get('train.csv').contain))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))

    @unittest.skip("not working")
    def test_ast_for_node_target_multiple_variables(self):
        value = "for train_index, test_index in kf:\n" \
                "\tprint(train_index)\n" \
                "\tprint(test_index)"

        self.assertTrue(False)

    # TODO: VERIFY THIS TEST
    @unittest.skip("not working")
    def test_ast_for_node_call_all_files(self):
        value = "full_data = [train, test]\n" \
                "for dataset in full_data:\n" \
                "\tage_avg = dataset['Age'].mean()\n" \
                "\tage_std = dataset['Age'].std()\n" \
                "\tage_null_count = dataset['Age'].isnull().sum()"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)

        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor.files = {'train': File("train.csv"), 'test': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Age'])
        node_visitor.visit(tree)

        col_uri_train = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col_uri_test = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Age')

        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')
        mean_uri = util.create_import_uri('pandas.DataFrame.mean')
        std_uri = util.create_import_uri('pandas.DataFrame.std')
        isnull_uri = util.create_import_uri('pandas.DataFrame.isnull')
        sum_uri = util.create_import_uri('pandas.DataFrame.sum')

        self.assertEqual(1, len(self.graph.head.next.next.calls))
        self.assertEqual(1, len(self.graph.head.next.next.next.calls))
        self.assertEqual(1, len(self.graph.head.next.next.next.next.calls))
        self.assertEqual(1, len(self.graph.tail.calls))
        self.assertEqual(2, len(self.graph.head.next.next.read))
        self.assertEqual(col_uri_test, self.graph.head.next.next.read[0].uri)
        self.assertEqual(col_uri_train, self.graph.head.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.head.next.next.next.read))
        self.assertEqual(col_uri_test, self.graph.head.next.next.next.read[0].uri)
        self.assertEqual(col_uri_train, self.graph.head.next.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.head.next.next.next.next.read))
        self.assertEqual(col_uri_test, self.graph.head.next.next.next.next.read[0].uri)
        self.assertEqual(col_uri_train, self.graph.head.next.next.next.next.read[1].uri)


class VisitSubscriptNode(Test):
    @unittest.skip("not working")
    def test_ast_constant_node_return_string(self):
        self.assertEqual(True, False)

    def test_ast_subscript_type_is_passed_through_the_subscript(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "missing_counts = train.isnull().sum().sort_values(ascending = False)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.visit(tree)

        generate_id(self.graph.head)
        lib_uri1 = util.create_import_uri('pandas.DataFrame.isnull')
        lib_uri2 = util.create_import_uri('pandas.DataFrame.sum')
        lib_uri3 = util.create_import_uri('pandas.DataFrame.sort_values')

        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')

        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 2),
                         self.graph.head.next.uri)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 3),
                         self.graph.head.next.next.uri)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 4),
                         self.graph.head.next.next.next.uri)
        self.assertEqual(1, len(self.graph.head.next.calls))
        self.assertEqual(0, len(self.graph.head.next.parameters))
        self.assertEqual(1, len(self.graph.head.next.next.calls))
        self.assertEqual(5, len(self.graph.head.next.next.parameters))
        self.assertEqual(1, len(self.graph.head.next.next.next.calls))
        self.assertEqual(8, len(self.graph.head.next.next.next.parameters))
        self.assertEqual(lib_uri1, self.graph.head.next.calls[0].uri)
        self.assertEqual(lib_uri2, self.graph.head.next.next.calls[0].uri)
        self.assertEqual(lib_uri3, self.graph.head.next.next.next.calls[0].uri)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertEqual(2, len(self.graph.libraries.get(pandas_uri).contain))
        self.assertEqual(3, len(self.graph.libraries.get(pandas_uri).contain.get(dataframe_uri).contain))
        self.assertIn('missing_counts', node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get('missing_counts'),
                         pd_dataframe)

    def test_ast_subscript_return_types(self):
        value = "data['Age'] = data.groupby(['Pclass', 'Sex'])['Age']"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)

        node_visitor.files = {'data': File("train.csv")}
        node_visitor.variables = {'data': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Pclass', 'Sex', 'Age'])

        node_visitor.visit(tree)

        generate_id(self.graph.head)
        lib_uri = util.create_import_uri('pandas.DataFrame.groupby')
        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')

        self.assertEqual(value, self.graph.head.text)
        self.assertEqual(util.create_statement_uri(SOURCE, DATASET_NAME, FILENAME, 1),
                         self.graph.head.uri)
        self.assertEqual(4, len(self.graph.head.read))
        self.assertEqual(9, len(self.graph.head.parameters))
        self.assertEqual(1, len(self.graph.head.calls))

        self.assertEqual(util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age'),
                         self.graph.head.read.pop().uri)
        self.assertEqual(util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age'),
                         self.graph.head.read.pop().uri)
        self.assertEqual(util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Sex'),
                         self.graph.head.read.pop().uri)
        self.assertEqual(util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Pclass'),
                         self.graph.head.read.pop().uri)
        self.assertEqual(lib_uri, self.graph.head.calls[0].uri)
        self.assertEqual(lib_uri,
                         self.graph.libraries.get(pandas_uri).contain.get(dataframe_uri).contain.get(lib_uri).uri)

    @unittest.skip("not working")
    def test_ast_subscript_deals_with_string(self):  # Titanic_Survival_Method.py line: 360
        value = "i.replace('.', '').replace('/', '').strip().split(' ')[0]"

        self.assertTrue(False)

    def test_subscript_return_name_with_int_index(self):
        node = ast.parse('train[0]').body[0].value
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables['train'] = pd_dataframe
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['a'])
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.add_node('1')

        name, _ = node_visitor.visit_Subscript(node)
        self.assertEqual('train', name)
        self.assertEqual(0, len(node_visitor.columns))
        self.assertEqual(1, len(self.graph.tail.read))

    def test_subscript_return_name_with_str_index(self):
        node = ast.parse("train['a']").body[0].value
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables['train'] = pd_dataframe
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['a'])
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.add_node('1')

        name, _ = node_visitor.visit_Subscript(node)
        self.assertEqual('train', name)
        self.assertEqual(0, len(node_visitor.columns))
        self.assertEqual(1, len(self.graph.tail.read))


class VisitSliceNode(Test):
    def test_ast_slice_identify_columns_inside_the_slice(self):
        value = "alone = train[(train['SibSp'] == 0) & (train['Parch'] == 0)]"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))

        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['SibSp', 'Parch'])
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)
        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'SibSp')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Parch')

        self.assertEqual(2, len(self.graph.head.read))
        self.assertEqual(col2_uri, self.graph.head.read.pop().uri)
        self.assertEqual(col1_uri, self.graph.head.read.pop().uri)
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))

    def test_ast_slice_identify_columns_inside_slice_more_complicated(self):
        value = "value = train[train['Survived'] == 1]['Age'].dropna()"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived', 'Age'])
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)
        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        lib_uri = util.create_import_uri('pandas.DataFrame.dropna')
        pandas_uri = util.create_import_uri('pandas')
        dataframe_uri = util.create_import_uri('pandas.DataFrame')

        self.assertEqual(2, len(self.graph.head.read))
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(col2_uri, self.graph.head.read.pop().uri)
        self.assertEqual(col1_uri, self.graph.head.read.pop().uri)
        self.assertEqual(lib_uri, self.graph.head.calls.pop().uri)
        self.assertEqual(lib_uri,
                         self.graph.libraries.get(pandas_uri).contain.get(dataframe_uri).contain.get(lib_uri).uri)
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))

    def test_ast_slice_identify_columns_in_slice_containing_a_list(self):
        value = "value = train[['Age', 'Survived']]"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived', 'Age'])
        node_visitor.files = {'train': File("train.csv")}
        node_visitor.variables = {'train': pd_dataframe}

        node_visitor.visit(tree)
        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')

        self.assertEqual(2, len(self.graph.head.read))
        self.assertEqual(col2_uri, self.graph.head.read.pop().uri)
        self.assertEqual(col1_uri, self.graph.head.read.pop().uri)
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))

    @unittest.skip("not working")
    def test_ast_slice_verify_if_columns_in_index_tuple(self):
        value = "test_df.loc[666, 'GarageQual'] = 'TA'"

        self.assertFalse(True)


class VisitClassDefNode(Test):
    @unittest.skip("not working")
    def test_ast_class_save_class_in_memory_with_variables_and_subgraph(self):
        value = "class skew_dummies(BaseEstimator, TransformerMixin):\n" \
                "\tdef __init__(self,skew=0.5):\n" \
                "\t\tself.skew = skew\n\n" \
                "\tdef fit(self,X,y=None):\n" \
                "\t\treturn self\n\n" \
                "\tdef transform(self,X):\n" \
                "\t\tX_numeric=X.select_dtypes(exclude=['object'])\n" \
                "\t\tskewness = X_numeric.apply(lambda x: skew(x))\n" \
                "\t\tskewness_features = skewness[abs(skewness) >= self.skew].index\n" \
                "\t\tX[skewness_features] = np.log1p(X[skewness_features])\n" \
                "\t\tX = pd.get_dummies(X)\n" \
                "\t\treturn X\n"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.visit(tree)

        self.assertIn("skew_dummies", node_visitor.subgraph)
        self.assertIn('__init__', node_visitor.subgraph.get("skew_dummies").subgraph.keys())
        self.assertIn('fit', node_visitor.subgraph.get("skew_dummies").subgraph.keys())
        self.assertIn('transform', node_visitor.subgraph.get("skew_dummies").subgraph.keys())

    @unittest.skip
    def test_ast_class_save_variables_after_initialization(self):
        value = "class skew_dummies:\n" \
                "\tdef __init__(self,skew=0.5):\n" \
                "\t\tself.skew = skew\n\n" \
                "value = skew_dummies(1)"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.visit(tree)

        self.assertIn("value", node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get("value"), "skew_dummies")

    @unittest.skip("not working")
    def test_ast_class_keep_values_from_ancestors(self):
        value = "class DataFrameImputer(TransformerMixin):\n" \
                "\tdef fit(self, X, y=None):\n" \
                "\t\tself.fill = pd.Series([X[c].value_counts().index[0]\n" \
                "\t\t\tif X[c].dtype == np.dtype('O') else X[c].median() for c in X],\n" \
                "\t\t\tindex=X.columns)\n" \
                "\t\treturn self\n" \
                "\tdef transform(self, X, y=None):\n" \
                "\t\treturn X.fillna(self.fill)"

        self.assertTrue(False)


class VisitFunctionDef(Test):
    def test_ast_function_return_type_return_value(self):
        value = "def cleanData(data):\n" \
                "\treturn data\n" \
                "clean_data = cleanData(values)"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.variables = {"values": pd_dataframe}
        node_visitor.visit(tree)

        self.assertIn("cleanData", node_visitor.subgraph)
        self.assertIn("clean_data", node_visitor.variables.keys())
        self.assertEqual(node_visitor.variables.get("clean_data").name,
                         pd_dataframe.name)


class ColumnSeparation(Test):
    def test_column_associated_to_the_right_table(self):
        value = "file1 = pd.read_csv('file1.csv')\n" \
                "file2 = pd.read_csv('file2.csv')\n" \
                "col1 = file1['col1']\n" \
                "col2 = file2['col2']"
        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['file1.csv'] = pd.DataFrame(columns=['col1'])
        node_visitor.working_file['file2.csv'] = pd.DataFrame(columns=['col2'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file1.csv', 'col1')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file2.csv', 'col2')
        file1_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file1.csv')
        file2_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file2.csv')

        self.assertEqual(col1_uri, self.graph.head.next.next.read.pop().uri)
        self.assertEqual(col2_uri, self.graph.head.next.next.next.read.pop().uri)
        self.assertEqual(2, len(self.graph.files))
        self.assertIn(file1_uri, self.graph.files.get('file1.csv').uri)
        self.assertIn(file2_uri, self.graph.files.get('file2.csv').uri)
        self.assertEqual(1, len(self.graph.files.get('file1.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('file2.csv').contain))

    def test_column_associated_to_right_table_other(self):
        value = "file1 = pd.read_csv('file1.csv')\n" \
                "file2 = pd.read_csv('file2.csv')\n" \
                "col1 = file1.col1.sum()\n" \
                "col2 = file2.col2.sum()"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['file1.csv'] = pd.DataFrame(columns=['col1'])
        node_visitor.working_file['file2.csv'] = pd.DataFrame(columns=['col2'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file1.csv', 'col1')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file2.csv', 'col2')
        file1_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file1.csv')
        file2_uri = util.create_file_uri(SOURCE, DATASET_NAME, 'file2.csv')

        self.assertEqual(1, len(self.graph.head.next.next.read))
        self.assertEqual(1, len(self.graph.head.next.next.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.next.read.pop().uri)
        self.assertEqual(col2_uri, self.graph.head.next.next.next.read.pop().uri)
        self.assertEqual(2, len(self.graph.files))
        self.assertIn(file1_uri, self.graph.files.get('file1.csv').uri)
        self.assertIn(file2_uri, self.graph.files.get('file2.csv').uri)
        self.assertEqual(1, len(self.graph.files.get('file1.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('file2.csv').contain))

    def test_column_assign_correctly_to_file_in_loop(self):
        value = "file1 = pd.read_csv('file1.csv')\n" \
                "file2 = pd.read_csv('file2.csv')\n" \
                "for feature in ['Embarked', 'Survived']:\n" \
                "\tfile1[feature] = 1\n" \
                "\tfile2[feature] = 1"

        tree = ast.parse(value)
        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['file1.csv'] = pd.DataFrame(columns=['Embarked', 'Survived'])
        node_visitor.working_file['file2.csv'] = pd.DataFrame(columns=['Embarked', 'Survived'])
        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file1.csv', 'Embarked')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file1.csv', 'Survived')
        col3_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file2.csv', 'Embarked')
        col4_uri = util.create_column_name(SOURCE, DATASET_NAME, 'file2.csv', 'Survived')

        self.assertEqual(2, len(self.graph.head.next.next.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.next.next.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.head.next.next.next.next.read))
        self.assertEqual(col3_uri, self.graph.head.next.next.next.next.read[0].uri)
        self.assertEqual(col4_uri, self.graph.head.next.next.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.files.get('file1.csv').contain))
        self.assertEqual(2, len(self.graph.files.get('file2.csv').contain))
        self.assertIn(self.graph.files.get('file1.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('file1.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('file2.csv').contain.pop().uri, (col3_uri, col4_uri))
        self.assertIn(self.graph.files.get('file2.csv').contain.pop().uri, (col3_uri, col4_uri))

    def test_column_assign_correctly_inside_slice(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "test = pd.read_csv('test.csv')\n" \
                "alone = train[(train['SibSp'] == 0) & (train['Parch'] == 0)]\n" \
                "not_alone = test[(test['SibSp'] == 0) & (test['Parch'] == 0)]"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['SibSp', 'Parch'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['SibSp', 'Parch'])
        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'SibSp')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Parch')
        col3_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'SibSp')
        col4_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Parch')

        self.assertEqual(2, len(self.graph.head.next.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.next.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.head.next.next.next.read))
        self.assertEqual(col3_uri, self.graph.head.next.next.next.read[0].uri)
        self.assertEqual(col4_uri, self.graph.head.next.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(2, len(self.graph.files.get('test.csv').contain))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))

    def test_column_assign_correctly_in_more_complicated_slice(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "test = pd.read_csv('test.csv')\n" \
                "value = train[train['Survived'] == 1]['Age'].dropna()\n" \
                "value = test[test['Sex'] == 1]['Work'].dropna()"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived', 'Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Sex', 'Work'])
        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col3_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')
        col4_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Work')

        self.assertEqual(2, len(self.graph.head.next.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.next.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.head.next.next.next.read))
        self.assertEqual(col3_uri, self.graph.head.next.next.next.read[0].uri)
        self.assertEqual(col4_uri, self.graph.head.next.next.next.read[1].uri)
        self.assertEqual(2, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(2, len(self.graph.files.get('test.csv').contain))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))

    def test_column_assign_correctly_with_index(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "test = pd.read_csv('test.csv')\n" \
                "y = train[0]\n" \
                "z = test[0]"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias['pd'] = 'pandas'
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Sex'])
        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')

        self.assertEqual(1, len(self.graph.head.next.next.read))
        self.assertEqual(col1_uri, self.graph.head.next.next.read[0].uri)
        self.assertEqual(1, len(self.graph.head.next.next.next.read))
        self.assertEqual(col2_uri, self.graph.head.next.next.next.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('test.csv').contain))
        self.assertEqual(col1_uri, self.graph.files.get('train.csv').contain.pop().uri)
        self.assertEqual(col2_uri, self.graph.files.get('test.csv').contain.pop().uri)

    def test_column_assign_correctly_in_subscript(self):
        value = "data['Age'] = data.groupby(['Pclass', 'Sex'])['Age']\n" \
                "test['ge'] = test.groupby(['Plass', 'ex'])['ge']"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'test.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'data': File("train.csv"), 'test': File("test.csv")}
        node_visitor.variables = {'data': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Pclass', 'Sex', 'Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Plass', 'ex', 'ge'])

        node_visitor.visit(tree)

        self.assertEqual(4, len(self.graph.head.read))
        self.assertEqual(9, len(self.graph.head.parameters))
        self.assertEqual(1, len(self.graph.head.calls))

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Sex')
        col3_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Pclass')
        col4_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'ge')
        col5_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'ex')
        col6_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Plass')

        self.assertEqual(col3_uri, self.graph.head.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.read[1].uri)
        self.assertEqual(col1_uri, self.graph.head.read[2].uri)
        self.assertEqual(col6_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(col5_uri, self.graph.head.next.read[1].uri)
        self.assertEqual(col4_uri, self.graph.head.next.read[2].uri)
        self.assertEqual(4, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(4, len(self.graph.files.get('test.csv').contain))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri, col3_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri, col3_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri, col3_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col4_uri, col5_uri, col6_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col4_uri, col5_uri, col6_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col4_uri, col5_uri, col6_uri))

    def test_ast_slice_identify_columns_in_slice_containing_a_list(self):
        value = "value = train[['Age', 'Survived']]\n" \
                "z = test[['Ag', 'Sex']]"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'test.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv"), 'test': File("test.csv")}
        node_visitor.variables = {'train': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Survived', 'Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Ag', 'Sex'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Survived')
        col3_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Ag')
        col4_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')

        self.assertEqual(2, len(self.graph.head.read))
        self.assertEqual(2, len(self.graph.head.next.read))
        self.assertEqual(col1_uri, self.graph.head.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.read[1].uri)
        self.assertEqual(col3_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(col4_uri, self.graph.head.next.read[1].uri)
        self.assertEqual(2, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(2, len(self.graph.files.get('test.csv').contain))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('train.csv').contain.pop().uri, (col1_uri, col2_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))
        self.assertIn(self.graph.files.get('test.csv').contain.pop().uri, (col3_uri, col4_uri))

    def test_column_assign_correctly_in_keywords(self):
        value = "value = train.drop(labels='Age')\n" \
                "z = test.drop(labels='Sex')"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'test.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv"), 'test': File("test.csv")}
        node_visitor.variables = {'train': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Sex'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')

        self.assertEqual(1, len(self.graph.head.read))
        self.assertEqual(1, len(self.graph.head.next.read))
        self.assertEqual(col1_uri, self.graph.head.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('test.csv').contain))
        self.assertEqual(col1_uri, self.graph.files.get('train.csv').contain.pop().uri)
        self.assertEqual(col2_uri, self.graph.files.get('test.csv').contain.pop().uri)

    def test_column_assign_correctly_as_variable(self):
        value = "value = train.drop('Age')\n" \
                "z = test.drop('Sex')"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'test.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv"), 'test': File("test.csv")}
        node_visitor.variables = {'train': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Sex'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')

        self.assertEqual(1, len(self.graph.head.read))
        self.assertEqual(1, len(self.graph.head.next.read))
        self.assertEqual(col1_uri, self.graph.head.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('test.csv').contain))
        self.assertEqual(col1_uri, self.graph.files.get('train.csv').contain.pop().uri)
        self.assertEqual(col2_uri, self.graph.files.get('test.csv').contain.pop().uri)

    def test_column_assign_correctly_in_keywords(self):
        value = "value = train.drop('Age')\n" \
                "z = test.drop('Sex')"
        tree = ast.parse(value)
        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.graph.files['test.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'test.csv'))
        node_visitor = NodeVisitor(self.graph)
        node_visitor.files = {'train': File("train.csv"), 'test': File("test.csv")}
        node_visitor.variables = {'train': pd_dataframe, 'test': pd_dataframe}
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])
        node_visitor.working_file['test.csv'] = pd.DataFrame(columns=['Sex'])

        node_visitor.visit(tree)

        col1_uri = util.create_column_name(SOURCE, DATASET_NAME, 'train.csv', 'Age')
        col2_uri = util.create_column_name(SOURCE, DATASET_NAME, 'test.csv', 'Sex')

        self.assertEqual(1, len(self.graph.head.read))
        self.assertEqual(1, len(self.graph.head.next.read))
        self.assertEqual(col1_uri, self.graph.head.read[0].uri)
        self.assertEqual(col2_uri, self.graph.head.next.read[0].uri)
        self.assertEqual(1, len(self.graph.files.get('train.csv').contain))
        self.assertEqual(1, len(self.graph.files.get('test.csv').contain))
        self.assertEqual(col1_uri, self.graph.files.get('train.csv').contain.pop().uri)
        self.assertEqual(col2_uri, self.graph.files.get('test.csv').contain.pop().uri)


class Package(Test):
    def test_built_in_method(self):
        value = "a = len(['a', 'b'])"
        parse_and_visit_node(value, self.graph)

        lib_uri = util.create_built_in_uri("len")
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(lib_uri, self.graph.head.calls[0].uri)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertIn(lib_uri, self.graph.libraries.keys())

    def test_built_in_method_in_for_loop(self):
        value = "for i in range(10):\n" \
                "\tprint(i)"
        parse_and_visit_node(value, self.graph)

        lib_uri = util.create_built_in_uri("range")
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(lib_uri, self.graph.head.calls[0].uri)
        self.assertEqual(1, len(self.graph.libraries))
        self.assertIn(lib_uri, self.graph.libraries.keys())


class DataFlowTesting(Test):
    def test_basic_dataflow(self):
        value = "x = 1\n" \
                "y = x + 1"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_function_call(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = train.sum()"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_parameter_passing(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = pd.DataFrame(train)"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_keyword_passing(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = pd.DataFrame(data=train)"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_consecutive_function_call(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = train.isnull().sum()"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.head.next.data_flow))
        self.assertEqual(self.graph.head.next.next.uri,
                         self.graph.head.next.data_flow[0].uri)

    def test_more_complex_consecutive_function_call(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = train.isnull().sum().sort_values(ascending=False)"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.head.next.data_flow))
        self.assertEqual(self.graph.head.next.next.uri,
                         self.graph.head.next.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.tail.previous.data_flow))
        self.assertEqual(self.graph.tail.uri,
                         self.graph.tail.previous.data_flow[0].uri)

    def test_subscript(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = train['row']"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_user_function_call(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "def a(x):\n" \
                "\ty = x.sum()\n" \
                "p = a(train)"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.next.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.tail.previous.data_flow))

    def test_tuple_assign(self):
        value = "train, test = train_test_split()\n" \
                "x = train.sum()\n" \
                "y = test.sum()"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(2, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(self.graph.head.next.next.uri,
                         self.graph.head.data_flow[1].uri)

    def test_column_name_as_attribute(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "x = train.Age.sum()"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)

    def test_assign_to_subscript(self):
        value = "train = pd.read_csv('train.csv')\n" \
                "train['Age'] = 1\n" \
                "x = train.sum()"

        parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)

        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.head.next.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.head.next.data_flow))
        self.assertEqual(self.graph.tail.uri,
                         self.graph.head.next.data_flow[0].uri)

    def test_list(self):
        value = "model1 = pd.read_csv('1.csv')\n" \
                "model2 = pd.read_csv('2.csv')\n" \
                "model3 = pd.read_csv('3.csv')\n" \
                "models = [model1, model2, model3]"

        node_visitor = parse_and_visit_node(value, self.graph)
        generate_id(self.graph.head)
        self.assertEqual(1, len(self.graph.head.data_flow))
        self.assertEqual(self.graph.tail.uri,
                         self.graph.head.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.head.next.data_flow))
        self.assertEqual(self.graph.tail.uri,
                         self.graph.head.next.data_flow[0].uri)
        self.assertEqual(1, len(self.graph.head.next.next.data_flow))
        self.assertEqual(self.graph.tail.uri,
                         self.graph.head.next.next.data_flow[0].uri)


class TestLibraryPath(Test):
    def test_method_return_library_path_from_package_name(self):
        package1 = 'pd.read_csv'
        package2 = 'pd.isnull'
        package3 = 'print'

        node_visitor = NodeVisitor(self.graph)
        node_visitor.alias = {'pd': 'pandas'}

        self.assertEqual('pandas.read_csv', node_visitor._create_library_path(package1))
        self.assertEqual('pandas.isnull', node_visitor._create_library_path(package2))
        self.assertEqual('print', node_visitor._create_library_path(package3))

    def test_method_return_library_path_from_package_without_obvious_path(self):
        package1 = 'SVC'
        package2 = 'LabelEncoder'

        value = "from sklearn.svm import SVC \n" \
                "from sklearn.preprocessing import LabelEncoder"

        node_visitor = parse_and_visit_node(value, self.graph)

        self.assertEqual('sklearn.svm.SVC', node_visitor._create_library_path(package1))
        self.assertEqual('sklearn.preprocessing.LabelEncoder',
                         node_visitor._create_library_path(package2))

    def test_extract_dataflow(self):
        node_visitor = NodeVisitor(self.graph)
        node_visitor._extract_dataflow([])


class TestCreatePackageClass(Test):
    def test_None_package_should_not_call(self):
        node_visitor = NodeVisitor(self.graph)
        node_visitor.graph_info.add_node('')

        node_visitor._create_package_call(Calls.Call(is_relevant=False), None)
        self.assertEqual(0, len(self.graph.head.calls))
        self.assertEqual(0, len(self.graph.libraries))

    def test_not_watched_package_should_not_call(self):
        node_visitor = NodeVisitor(self.graph)
        node_visitor.graph_info.add_node('')

        node_visitor._create_package_call(Calls.Call(is_relevant=False), 'matplotlib.pyplot.show')
        self.assertEqual(0, len(self.graph.head.calls))
        self.assertEqual(0, len(self.graph.libraries))

    def test_watched_package_should_call(self):
        node_visitor = NodeVisitor(self.graph)
        node_visitor.graph_info.add_node('')

        node_visitor._create_package_call(pd_dataframe, 'pd.DataFrame')
        self.assertEqual(1, len(self.graph.head.calls))
        self.assertEqual(1, len(self.graph.libraries))

    def test_weird_aspect_of_kglids(self):
        tree = 'import matplotlib.pyplot as plt\n' \
               'plt.show()'

        parse_and_visit_node(tree, self.graph)
        for l in self.graph.libraries.values():
            print(l.uri)
        # print(self.graph.libraries.values())


class TestFileContainingElement(Test):
    def test_file_containing_column_in_set(self):
        value = "data = pandas.read_csv('train.csv')\n" \
                "data['Age'] = data.groupby(['Pclass', 'Sex'])['Age']"
        tree = ast.parse(value)

        self.graph.files['train.csv'] = DataTypes.File(util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        node_visitor = NodeVisitor(self.graph)

        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Pclass', 'Sex', 'Age'])

        node_visitor.visit(tree)

        self.assertNotEqual(0, len(self.graph.files.get('train.csv', DataTypes.File('')).contain))

    def test_column_read_in_sub_graph(self):
        value = "df = pandas.read_csv('train.csv')\n" \
                "x = df.groupby('A')['B']"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['A', 'B'])

        node_visitor.visit(tree)
        self.assertEqual(2, len(self.graph.tail.read))

    def test_column_read_in_sub_graph(self):
        value = "df = pandas.read_csv('train.csv')\n" \
                "x = df[1:]"
        tree = ast.parse(value)

        node_visitor = NodeVisitor(self.graph)
        node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['A', 'B'])

        node_visitor.visit(tree)
        self.assertEqual(2, len(self.graph.tail.read))

    def test_column_read_in_sub_graph(self):
        value = "x = pandas.read_csv('train.csv')\n" \
                "x = x[1:]"
        tree = ast.parse(value).body[0].value
        print(tree.__dict__)
        parse_and_visit_node(value, self.graph)

    def test_random_thing(self):
        value = "print(pandas.DataFrame(columns=['a']))"
        node_visitor = parse_and_visit_node(value, self.graph)

        print(self.graph.head.text)
        print(self.graph.head.calls[0].uri)
        self.assertEqual(1, 1)

    def test_insert_parameters(self):
        value = "x = sklearn.svm.SVC(**ssss)"
        parse_and_visit_node(value, self.graph)
        for edge in self.graph.tail.parameters:
            print(edge.str())

if __name__ == '__main__':
    unittest.main()
