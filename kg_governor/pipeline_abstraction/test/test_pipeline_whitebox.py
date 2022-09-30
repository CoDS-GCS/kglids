import unittest
import ast
import pandas as pd

from src.datatypes import File as DataFile, GraphInformation
from src.ast_package import (Name, Attribute, Constant, Call, List, Dict, Subscript, Lambda, BinOp, Tuple,
                             get_ast_package, Compare)
from src.ast_package.types import CallComponents, CallArgumentsComponents, AssignComponents, BinOpComponents, \
    AttributeComponents
from src.Calls import packages, pd_dataframe, File
from src.pipeline_abstraction import NodeVisitor
import src.util as util

kglids_library = "http://kglids.org/pipeline/library/"
FILENAME = "test.py"
SOURCE = "<SOURCE>"
DATASET_NAME = "<DATASET_NAME>"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        graph = GraphInformation(
            python_file_name=FILENAME,
            source=SOURCE,
            dataset_name=DATASET_NAME
        )
        self.node_visitor = NodeVisitor(graph)


class AstPackageSelection(Test):
    def test_get_ast_package_return_Name_with_ast_name(self):
        name = ast.Name()
        ast_package = get_ast_package(name)

        self.assertEqual(type(Name()), type(ast_package))

    def test_get_ast_package_return_Attribute_with_ast_attribute(self):
        attr = ast.Attribute()
        ast_package = get_ast_package(attr)

        self.assertEqual(type(Attribute()), type(ast_package))

    def test_get_ast_package_return_Constant_with_ast_constant(self):
        constant = ast.Constant()
        ast_package = get_ast_package(constant)

        self.assertEqual(type(Constant()), type(ast_package))

    def test_get_ast_package_return_Call_with_ast_call(self):
        call = ast.Call()
        ast_package = get_ast_package(call)

        self.assertEqual(type(Call()), type(ast_package))

    def test_get_ast_package_return_List_with_ast_list(self):
        a_list = ast.List()
        ast_package = get_ast_package(a_list)

        self.assertEqual(type(List()), type(ast_package))

    def test_get_ast_package_return_Dict_with_ast_dict(self):
        a_dict = ast.Dict()
        ast_package = get_ast_package(a_dict)

        self.assertEqual(type(Dict()), type(ast_package))

    def test_get_ast_package_return_Subscript_with_ast_subscript(self):
        subscript = ast.Subscript()
        ast_package = get_ast_package(subscript)

        self.assertEqual(type(Subscript()), type(ast_package))

    def test_get_ast_package_return_Lambda_with_ast_lambda(self):
        a_lambda = ast.Lambda()
        ast_package = get_ast_package(a_lambda)

        self.assertEqual(type(Lambda()), type(ast_package))

    def test_get_ast_package_return_BinOp_with_ast_binOp(self):
        binop = ast.BinOp()
        ast_package = get_ast_package(binop)

        self.assertEqual(type(BinOp()), type(ast_package))

    def test_get_ast_package_return_Tuple_with_ast_tuple(self):
        a_tuple = ast.Tuple()
        ast_package = get_ast_package(a_tuple)

        self.assertEqual(type(Tuple()), type(ast_package))

    def test_get_ast_package_return_Compare_with_ast_compare(self):
        compare = ast.Compare()
        ast_package = get_ast_package(compare)

        self.assertEqual(type(Compare()), type(ast_package))


class ClassNodeTest(Test):
    # def test_keep_track_of_class_variable_name(self):
    #     value = 'class Test():\n' \
    #            '\tdef __init__():\n' \
    #            '\t\tx = 1'
    #
    #     # node_visitor = NodeVisitor(self.graph)
    #     tree = ast.parse(value)
    #     node_visitor.visit(tree)

    def test_save_functions_and_link_them_to_variable(self):
        pass

    def test_init_save_variables_within_sub_node_visitor(self):
        pass

    def test_class_node_visitor_can_access_parent_node_visitor_variables_etc(self):
        pass


class AstNameTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.name = Name()

    def test_extract_func(self):
        value = "model5 = SVC()"
        components = CallComponents()
        call = ast.parse(value).body[0].value

        self.name.extract_func(self.node_visitor, call, components)

        self.assertEqual('SVC', components.package)

    def test_analyze_call_arguments_to_extract_argument(self):
        value = "pd.DataFrame(train)"
        self.node_visitor.files['train'] = 'train.csv'
        self.node_visitor.variables['train'] = 'here'
        call = ast.parse(value).body[0].value
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()

        result = self.name.analyze_call_arguments(self.node_visitor, call, arg_components, call_components, 0)

        self.assertEqual('here', result, 'argument does not match')
        self.assertEqual('train.csv', call_components.file, 'argument linked file is missing')
        self.assertEqual('train', arg_components.file_args[0], 'argument component missing linked file')

    def test_analyze_call_arguments_to_extract_argument_as_list(self):
        value = "pandas.read_csv(boat)"
        self.node_visitor.files['train'] = 'train.csv'
        self.node_visitor.variables['boat'] = ['train']
        call = ast.parse(value).body[0].value
        arg_components = CallArgumentsComponents(packages.get('pandas.read_csv').parameters.keys())
        call_components = CallComponents()

        result = self.name.analyze_call_arguments(self.node_visitor, call, arg_components, call_components, 0)

    #     TODO: FINISH THIS

    def test_extract_assign_value(self):
        value = "train = boat"
        assign = ast.parse(value).body[0]
        components = AssignComponents()

        self.name.extract_assign_value(self.node_visitor, assign, components)

        self.assertEqual('boat', components.value)

    def test_analyze_assign_target(self):
        value = "train = pd.read_csv('train.csv')"
        self.node_visitor.graph_info.add_node("TARGET")
        assign = ast.parse(value).body[0]
        components = AssignComponents()

        self.name.analyze_assign_target(self.node_visitor, assign.targets[0], components)

        self.assertEqual("TARGET", self.node_visitor.data_flow_container['train'].text)

    def test_extract_list_element(self):
        value = '[value]'
        a_list = ast.parse(value).body[0].value

        elements = []
        self.name.extract_list_element(self.node_visitor, a_list, 0, elements)

        self.assertEqual('value', elements[0])

    def test_extract_subscript_value(self):
        value = "train['a']"
        subscript = ast.parse(value).body[0].value

        result, _ = self.name.extract_subscript_value(self.node_visitor, subscript)

        self.assertEqual('train', result)

    def test_analyze_attribute_value(self):
        value = 'train.Age'
        self.node_visitor.variables['train'] = pd_dataframe
        attribute = ast.parse(value).body[0].value
        components = AttributeComponents()

        self.name.analyze_attribute_value(self.node_visitor, attribute, components)

        self.assertEqual('pandas.DataFrame.Age', components.path)
        self.assertEqual(None, components.file)
        self.assertEqual('train', components.parent_path)

    def test_analyze_attribute_when_column_in_name_return_without_column(self):
        value = 'train.Age'
        self.node_visitor.variables['train'] = pd_dataframe
        self.node_visitor.files['train'] = File("train.csv")
        self.node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])

        attribute = ast.parse(value).body[0].value
        components = AttributeComponents()
        self.name.analyze_attribute_value(self.node_visitor, attribute, components)

        self.assertEqual('pandas.DataFrame', components.path)
        self.assertIsNone(components.file)
        self.assertEqual('train', components.parent_path)


class AstAttributeTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.attribute = Attribute()

    def test_extract_func(self):
        value = "test.sum()"
        components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        self.attribute.extract_func(self.node_visitor, call_pkg, components)

        self.assertEqual('test', components.base_package, 'base package does not match')
        self.assertEqual('test.sum', components.package, 'package does not match')  # TODO: VERIFY THIS BEHAVIOR

    def test_extract_assign_value(self):
        value = 'result = name.sum'
        components = CallComponents
        assign_pkg = ast.parse(value).body[0]

        self.attribute.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertEqual('name.sum', components.value)

    def test_extract_subscript_value(self):
        value = "train.iloc[1, 2]"
        self.node_visitor.variables['train'] = pd_dataframe
        subscript = ast.parse(value).body[0].value

        result, _ = self.attribute.extract_subscript_value(self.node_visitor, subscript)

        self.assertEqual('pandas.DataFrame', result)

    def test_analyze_attribute_value(self):
        value = 'pandas.DataFrame.Age'
        attribute = ast.parse(value).body[0].value
        components = AttributeComponents()

        self.attribute.analyze_attribute_value(self.node_visitor, attribute, components)

        self.assertEqual('pandas.DataFrame.Age', components.path)
        self.assertEqual('pandas', components.parent_path)  # Note: Is this the way I want the export the value?
        self.assertIsNone(components.file)


class AstConstantTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.constant = Constant()

    def test_analyze_call_arguments(self):
        value = "pd.DataFrame('train')"
        self.node_visitor.files['train'] = 'train.csv'
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        value = self.constant.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertEqual('train', value, 'constant value does not match')
        self.assertEqual('train.csv', call_components.file, 'argument linked file is missing')
        self.assertEqual('train', arg_components.file_args[0], 'argument component missing linked file')

    def test_extract_assign_value(self):
        value = "a = 'value'"
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.constant.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertEqual('value', components.value, 'assigned value does not match')

    def test_extract_list_element(self):
        value = '["a"]'
        a_list = ast.parse(value).body[0].value

        elements = []
        self.constant.extract_list_element(self.node_visitor, a_list, 0, elements)

        self.assertEqual('a', elements[0])

    def test_analyze_bin_op_branch(self):
        value = "1 - 1"
        self.node_visitor.variables = {'df': pd_dataframe}

        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.constant.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertIsNone(component.left)


class AstCallTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.call = Call()

    def test_analyze_call_arguments(self):
        value = "print(pandas.DataFrame('model'))"
        self.node_visitor.files['train'] = 'train.csv'
        self.node_visitor.graph_info.add_node('2nd node')
        self.node_visitor.graph_info.add_node('1st node')
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        value = self.call.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertEqual('DataFrame', value, 'return type is not correct')

    def test_extract_assign_value(self):
        value = 'a = pandas.DataFrame("name")'
        self.node_visitor.graph_info.add_node('pandas.DataFrame("name"')
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.call.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertEqual('DataFrame', components.value[0].name, 'assign component value type is not correct')
        self.assertIsNone(components.file, 'assign component file should be empty')

    def test_extract_list_element(self):
        value = '[pandas.DataFrame()]'
        self.node_visitor.graph_info.add_node('node')
        a_list = ast.parse(value).body[0].value

        elements = []
        self.call.extract_list_element(self.node_visitor, a_list, 0, elements)

        self.assertEqual('pandas.DataFrame', elements[0])

    def test_analyze_bin_op_branch(self):
        value = "pandas.DataFrame.sum() - 1"
        self.node_visitor.variables = {'df': pd_dataframe}
        self.node_visitor.graph_info.add_node('1')

        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.call.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertEqual('pandas.DataFrame', component.left.full_path())

    def test_extract_subscript_value(self):
        value = "df.sum()['ER']"
        self.node_visitor.variables['df'] = pd_dataframe
        self.node_visitor.graph_info.add_node('1')
        subscript = ast.parse(value).body[0].value

        values, _ = self.call.extract_subscript_value(self.node_visitor, subscript)
        result, _ = values
        self.assertEqual('pandas.DataFrame', result)

    def test_analyze_attribute_value(self):
        value = 'train.sum().Age'
        self.node_visitor.graph_info.add_node('!')
        self.node_visitor.variables['train'] = pd_dataframe
        self.node_visitor.files['train'] = File('train.csv')
        self.node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['Age'])
        attribute = ast.parse(value).body[0].value
        components = AttributeComponents()

        self.call.analyze_attribute_value(self.node_visitor, attribute, components)

        self.assertEqual('pandas.DataFrame', components.path)
        self.assertEqual('train', components.parent_path)
        self.assertIsNone(components.file)


class AstListTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.list = List()

    def test_analyze_call_arguments(self):
        value = "pandas.DataFrame(['Pclass', 'Sex'])"
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        self.node_visitor.files['Pclass'] = 'train.csv'
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        result = self.list.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertListEqual(['Pclass', 'Sex'], result, 'the extract list is not correct')
        self.assertEqual('train.csv', call_components.file, 'File does not match with saved value')

    def test_extract_assign_value(self):
        value = 'a = ["a"]'
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.list.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertListEqual(['a'], components.value)

    def test_extract_list_element(self):
        value = '[["a"]]'
        a_list = ast.parse(value).body[0].value

        elements = []
        self.list.extract_list_element(self.node_visitor, a_list, 0, elements)

        self.assertListEqual(['a'], elements[0])


class AstDictTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.dict = Dict()

    def test_analyze_call_arguments(self):
        value = "pandas.DataFrame({'a': 1, 'b': 2})"
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        result = self.dict.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertDictEqual({'a': 1, 'b': 2}, result)

    def test_extract_assign_value(self):
        value = "a = {'a': 1, 'b': 2}"
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.dict.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertDictEqual({'a': 1, 'b': 2}, components.value)

    # def test_extract_list_element(self):
    #     value = '[{"a": 1}]'
    #     a_list = ast.parse(value).body[0].value
    #
    #     elements = []
    #     self.dict.extract_list_element(self.node_visitor, a_list, 0, elements)
    #
    #     self.assertDictEqual({'a': 1}, elements[0])


class AstSubscriptTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.subscript = Subscript()

    def test_analyze_call_arguments(self):
        value = "pandas.DataFrame(df[col])"
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        result = self.subscript.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertIsNone(result)

    def test_extract_assign_value(self):
        value = "a = df['col']"
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.node_visitor.variables['df'] = pd_dataframe
        self.node_visitor.files['df'] = File('train.csv')
        self.node_visitor.graph_info.files['train.csv'] = DataFile(
            util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['col'])
        self.node_visitor.graph_info.add_node('1')

        self.subscript.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertEqual('df', components.value)
        self.assertEqual(pd_dataframe.name, components.variable.name)
        self.assertEqual('train.csv', components.file.filename)
        self.assertEqual(1, len(self.node_visitor.graph_info.tail.read))

    def test_analyze_assign_target(self):
        value = "df['col'] = test"
        assign = ast.parse(value).body[0]
        components = AssignComponents()

        self.node_visitor.variables['df'] = pd_dataframe
        self.node_visitor.files['df'] = File('train.csv')
        self.node_visitor.graph_info.files['train.csv'] = DataFile(
            util.create_file_uri(SOURCE, DATASET_NAME, 'train.csv'))
        self.node_visitor.working_file['train.csv'] = pd.DataFrame(columns=['col'])
        self.node_visitor.graph_info.add_node('tail')

        result = self.subscript.analyze_assign_target(self.node_visitor, assign.targets[0], components)

        self.assertEqual('df', result, 'target should match')
        self.assertIsNotNone(self.node_visitor.data_flow_container['df'], 'dataflow should not be empty')
        self.assertEqual('tail', self.node_visitor.data_flow_container['df'].text, 'dataflow should be current node')
        self.assertEqual(1, len(self.node_visitor.graph_info.tail.read))

    def test_extract_list_element(self):
        value = '[df["col"]]'
        a_list = ast.parse(value).body[0].value

        elements = []
        self.subscript.extract_list_element(self.node_visitor, a_list, 0, elements)

        self.assertEqual('df', elements[0])

    def test_analyze_bin_op_branch_with_variable_saved_return_package(self):
        value = "df['a'] - 1"
        self.node_visitor.variables = {'df': pd_dataframe}

        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.subscript.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertEqual('pandas.DataFrame', component.left.full_path())

    def test_analyze_bin_op_branch(self):
        value = "df['a'] - 100"

        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.subscript.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertEqual('df', component.left)

    def test_analyze_bin_op_branch_when_calls_is_not_known_return(self):
        value = 'pandas.DataFrame.iloc[i, j] + deaths_group_df.iloc[i, j]'
        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.subscript.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertEqual('pandas.DataFrame', component.left)


class AstLambdaTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.ast_lambda = Lambda()

    def test_analyze_call_arguments(self):
        value = "map(lambda x: x + 1)"
        arg_components = CallArgumentsComponents(packages.get('pandas.DataFrame').parameters.keys())
        call_components = CallComponents()
        call_pkg = ast.parse(value).body[0].value

        result = self.ast_lambda.analyze_call_arguments(self.node_visitor, call_pkg, arg_components, call_components, 0)

        self.assertEqual('(lambda x: x + 1)', result)


class AstBinOpTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.bin_op = BinOp()

    def test_extract_assign_value(self):
        value = "y = x + 1"
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.bin_op.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertEqual("(x + 1)", components.value)

    def test_analyze_bin_op_branch(self):
        value = "pandas.DataFrame.sum() / 50 * 100"
        self.node_visitor.graph_info.add_node('1')

        component = BinOpComponents()
        branch = ast.parse(value).body[0].value
        self.bin_op.analyze_bin_op_branch(self.node_visitor, branch, 'left', component)

        self.assertEqual('pandas.DataFrame', component.left)


class AstTupleTest(Test):
    def setUp(self) -> None:
        super().setUp()
        self.tuple = Tuple()

    def test_extract_assign_value(self):
        value = "a = (1, 2, 3)"
        components = AssignComponents()
        assign_pkg = ast.parse(value).body[0]

        self.tuple.extract_assign_value(self.node_visitor, assign_pkg, components)

        self.assertListEqual([1, 2, 3], components.value)

    def test_analyze_assign_target(self):
        value = "a, b = train_test_split()"
        components = AssignComponents()
        self.node_visitor.graph_info.add_node("current")
        assign_pkg = ast.parse(value).body[0]

        self.tuple.analyze_assign_target(self.node_visitor, assign_pkg.targets[0], components)

        self.assertEqual(2, len(self.node_visitor.data_flow_container),
                         "data flow container is missing variable values")
        self.assertEqual('current', self.node_visitor.data_flow_container['a'].text,
                         'first variable should point to current node')
        self.assertEqual('current', self.node_visitor.data_flow_container['b'].text,
                         'second variable should point to current node')


if __name__ == '__main__':
    unittest.main()
