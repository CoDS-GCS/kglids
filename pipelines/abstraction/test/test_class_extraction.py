import unittest
import ast
import pandas as pd
from src.datatypes import GraphInformation, File as DataFile
from src.pipeline_abstraction import NodeVisitor
from src.ast_package import (Name, Attribute, Constant, Call, List, Dict, Subscript, Lambda, BinOp, Tuple,
                             get_ast_package, Compare)
from src.ast_package.types import CallComponents, CallArgumentsComponents, AssignComponents, BinOpComponents, \
    AttributeComponents
from Calls import packages, pd_dataframe, File
import util

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


class TestClass(Test):
    def test_class_when_created_saved_class_information_to_class_variables(self):
        value = "class MyClass:\n" \
                "\tdef __init__(self):\n" \
                "\t\tself.x = 1\n\n" \
                "\tdef myDef(self):" \
                "\tpass"
        tree = ast.parse(value)

        print(tree.__dict__)
        print(tree.body[0].__dict__)
        for fct in tree.body[0].body:
            print(fct.__dict__)
            if '__init__' in fct.name:
                print('init')
        # self.node_visitor.visit(tree)
        #
        # self.assertEqual(self.)

    def test_class_init_then_saved_variable_with_class_information(self):
        value = "class MyClass:\n" \
                "\tdef __init__(self):\n" \
                "\t\tself.x = 1"

        value = "x = MyClass()"

        a_class = ast.parse(value).body[0].value

        print(a_class.__dict__)
