import ast
import os
import astor
from _ast import (withitem, alias, keyword, arg, arguments, ExceptHandler, comprehension, NotIn, NotEq, LtE, Lt, IsNot,
                  Is, In, GtE, Gt, Eq, USub, UAdd, Not, Invert, Sub, RShift, Pow, MatMult, Mult, Mod, LShift, FloorDiv,
                  Div, BitXor, BitOr, BitAnd, Add, Or, And, Store, Load, Del, Tuple, List, Name, Starred, Subscript,
                  Attribute, NamedExpr, Constant, JoinedStr, FormattedValue, Call, Compare, YieldFrom, Yield, Await,
                  GeneratorExp, DictComp, SetComp, ListComp, Set, Dict, IfExp, Lambda, UnaryOp, BinOp, BoolOp, Slice,
                  Continue, Break, Pass, Expr, Nonlocal, Global, ImportFrom, Import, Assert, Try, Raise, AsyncWith,
                  With, If, While, AsyncFor, For, AnnAssign, AugAssign, Assign, Delete, Return, ClassDef,
                  AsyncFunctionDef, FunctionDef, Expression, Interactive, AST)
from ast import NameConstant, Bytes, Str, Num, Param, AugStore, AugLoad, Suite, Index, ExtSlice
from typing import Any
from collections import deque

import pandas as pd

import Calls
from src.datatypes import GraphInformation
from Calls import File, pd_dataframe, packages
from util import is_file, ControlFlow, get_package_info


def _format_node_text(node) -> str:
    text = astor.to_source(node).strip()
    if isinstance(node, ast.For):
        text = text.split("\n")[0]
    return text


def _insert_parameter(parameters: dict, is_block: bool, parameter: str, value):
    if is_block:
        parameters[parameter].append(value)
    else:
        parameters[parameter] = value


class NodeVisitor(ast.NodeVisitor):
    graph_info: GraphInformation

    # TODO: REWORK VARIABLE NAMING
    def __init__(self, graph_information: GraphInformation = None):
        self.graph_info = graph_information
        self.columns = []
        self.files = {}
        self.variables = {}
        self.alias = {}
        self.packages = {}
        self.subgraph = {}
        self.subgraph_node = {}
        self.control_flow = deque()
        self.user_defined_class = []
        self.working_file = {}
        self.target_node = None
        self.data_flow_container = {}
        self.library_path = {}  # Save library path that can't be extrapolated. i.e. train_test_split

    def visit(self, node: AST) -> Any:
        if type(node) in [ast.Assign, ast.Import, ast.ImportFrom, ast.Expr, ast.For, ast.If,
                          ast.FunctionDef, ast.AugAssign, ast.Call, ast.Return, ast.Attribute]:  # TODO: MaKE THIS PRETTIER
            self.columns.clear()
            self.graph_info.add_node(_format_node_text(node))
            if len(self.control_flow) > 0:
                self.graph_info.add_control_flow(self.control_flow)
        return super().visit(node)

    def visit_Interactive(self, node: Interactive) -> Any:
        pass

    def visit_Expression(self, node: Expression) -> Any:
        pass

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        subgraph = SubGraph()
        subgraph.graph_info = self.graph_info
        args = self.visit_arguments(node.args)
        subgraph.arguments = {i: a for a in args for i in range(len(args))}
        subgraph.working_file = self.working_file

        self.subgraph[node.name] = subgraph
        self.subgraph_node[node.name] = node

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> Any:
        pass

    def visit_ClassDef(self, node: ClassDef) -> Any:
        class_subgraph = ClassSubGraph()
        class_subgraph.graph_info = self.graph_info
        class_subgraph.working_file = self.working_file
        for base in node.bases:
            if isinstance(base, ast.Name):
                print(self.visit_Name(base))
            else:
                print("CLASS_DEF BASE:", node.__dict__)
                print(astor.to_source(node))

        el: ast.FunctionDef

    def visit_Return(self, node: Return) -> Any:
        pass

    def visit_Delete(self, node: Delete) -> Any:
        pass

    def visit_Assign(self, node: Assign) -> Any:
        # TODO: differentiate between return type and elements
        file = variable = None
        value = ''  # TODO: REFACTOR TYPE TO ARRAY
        if isinstance(node.value, ast.Constant):
            value = self.visit_Constant(node.value)
            file = self._file_creation(value)
        elif isinstance(node.value, ast.Call):
            return_types, file, _, _ = self.visit_Call(node.value)
            value = return_types
        elif isinstance(node.value, ast.Subscript):
            value = self.visit_Subscript(node.value)
            self._extract_dataflow(value)

            if not isinstance(value, list) and value in self.files.keys() and len(self.columns) > 0:
                file = self.files.get(value)
                if file is not None:
                    self.graph_info.add_columns(file.filename, self.columns)
                    self.columns.clear()
        elif isinstance(node.value, ast.List):
            value = self.visit_List(node.value)
            for el in value:
                self._extract_dataflow(el)
        elif isinstance(node.value, ast.BinOp):
            value = self.visit_BinOp(node.value)
        elif isinstance(node.value, ast.Name):
            value = self.visit_Name(node.value)
        elif isinstance(node.value, ast.Dict):
            value = self.visit_Dict(node.value)
        elif isinstance(node.value, ast.Attribute):
            value, base = self.visit_Attribute(node.value)
            file = self.files.get(base)
            if file is not None:
                self.graph_info.add_columns(file.filename, self.columns)
        elif isinstance(node.value, ast.Tuple):
            value = self.visit_Tuple(node.value)
        else:
            print("ASSIGN VALUE:", node.__dict__)
            print(astor.to_source(node))

        # NOTE: Extract target to assign
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = self.visit_Name(target)
                self.data_flow_container[name] = self.graph_info.tail
                if type(value) == str:
                    if file is not None:
                        self.files[name] = file
                    elif value in self.files.keys():
                        self.files[name] = self.files.get(value)
                    if value in self.variables.keys():
                        self.variables[name] = self.variables.get(value)
                elif isinstance(value, list):
                    if len(value) > 0:
                        self.variables[name] = value[0]
                    if file is not None:
                        self.files[name] = file
            elif isinstance(target, ast.Tuple):
                tuple_values = self.visit_Tuple(target)
                for el in tuple_values:
                    self.data_flow_container[el] = self.graph_info.tail

                if value is not None and tuple_values is not None:
                    for sub_target, package in tuple(zip(self.visit_Tuple(target), value)):
                        self.variables[sub_target] = package
                        if file is not None:
                            self.files[sub_target] = file

            elif isinstance(target, ast.Subscript):
                variable = self.visit_Subscript(target)
                self._extract_dataflow(variable)
                if isinstance(variable, str):
                    self.data_flow_container[variable] = self.graph_info.tail

        if variable is not None and not isinstance(variable, list):
            file = self.files.get(variable)
            if file is not None and len(self.columns) > 0:
                self.graph_info.add_columns(file.filename, self.columns)
                self.columns.clear()

    def visit_AugAssign(self, node: AugAssign) -> Any:
        pass

    def visit_AnnAssign(self, node: AnnAssign) -> Any:
        pass

    def visit_For(self, node: For) -> Any:
        if len(self.control_flow) == 0:
            self.graph_info.add_control_flow([ControlFlow.LOOP])
        fsg = ForSubGraph()
        fsg.graph_info = self.graph_info
        fsg.working_file = self.working_file
        fsg.files = self.files.copy()
        fsg.variables = self.variables.copy()
        fsg.packages = self.packages
        fsg.alias = self.alias
        fsg.subgraph = self.subgraph
        fsg.subgraph_node = self.subgraph_node
        fsg.control_flow = self.control_flow.copy()
        fsg.control_flow.append(ControlFlow.LOOP)
        fsg.data_flow_container = self.data_flow_container.copy()
        fsg.visit(node)

    def visit_AsyncFor(self, node: AsyncFor) -> Any:
        pass

    def visit_While(self, node: While) -> Any:
        pass

    def visit_If(self, node: If) -> Any:
        if len(self.control_flow) == 0:
            self.graph_info.add_control_flow([ControlFlow.CONDITIONAL])
        self.control_flow.append(ControlFlow.CONDITIONAL)

        for el in node.body:
            self.visit(el)
        for el in node.orelse:
            self.visit(el)

        self.control_flow.pop()

    def visit_With(self, node: With) -> Any:
        pass

    def visit_AsyncWith(self, node: AsyncWith) -> Any:
        pass

    def visit_Raise(self, node: Raise) -> Any:
        pass

    def visit_Try(self, node: Try) -> Any:
        pass

    def visit_Assert(self, node: Assert) -> Any:
        pass

    def visit_Import(self, node: Import) -> Any:
        self.graph_info.add_control_flow([ControlFlow.IMPORT])
        for import_name in node.names:
            name, as_name = self.visit_alias(import_name)

            if as_name is not None:
                self.alias[as_name] = name

            self.graph_info.add_import_node(name)

    def visit_ImportFrom(self, node: ImportFrom) -> Any:
        self.graph_info.add_control_flow([ControlFlow.IMPORT])
        path = node.module

        for lib_name in node.names:
            name, as_name = self.visit_alias(lib_name)
            if as_name is not None:
                self.alias[as_name] = name
            self.graph_info.add_import_from_node(path, name)
            self.library_path[name] = f'{path}.{name}'

    def visit_Global(self, node: Global) -> Any:
        pass

    def visit_Nonlocal(self, node: Nonlocal) -> Any:
        pass

    def visit_Expr(self, node: Expr) -> Any:
        if isinstance(node.value, ast.Call):
            self.visit_Call(node.value)
        elif isinstance(node.value, ast.Constant):
            pass
        else:
            print("EXPR VALUE:", node.__dict__)
            print(astor.to_source(node))

    def visit_Pass(self, node: Pass) -> Any:
        pass

    def visit_Break(self, node: Break) -> Any:
        pass

    def visit_Continue(self, node: Continue) -> Any:
        pass

    def visit_Slice(self, node: Slice) -> Any:
        pass

    def visit_BoolOp(self, node: BoolOp) -> Any:
        pass

    def visit_BinOp(self, node: BinOp) -> Any:
        left = ''
        right = ''
        if isinstance(node.left, ast.Constant):
            left = self.visit_Constant(node.left)
        elif isinstance(node.left, ast.Subscript):
            left = self.visit_Subscript(node.left)
        elif isinstance(node.left, ast.BinOp):
            left = self.visit_BinOp(node.left)
        elif isinstance(node.left, ast.Compare):
            self.visit_Compare(node.left)
        elif isinstance(node.left, ast.Call):
            subgraph = self._param_subgraph_init()
            subgraph.target_node = self.graph_info.tail if self.target_node is None else self.target_node
            left, _, _, _ = subgraph.visit(node.left)
            if left is not None and len(left) > 0:
                left = left[0]
        elif isinstance(node.left, ast.Name):
            name = self.visit_Name(node.left)
            self._extract_dataflow(name)
        else:
            print("LEFT VALUE:", node.__dict__)
            print(astor.to_source(node))

        if isinstance(node.right, ast.Call):
            psg = self._param_subgraph_init()
            psg.target_node = self.graph_info.tail if self.target_node is None else self.target_node
            right_type, _, _, _ = psg.visit(node.right)
        elif isinstance(node.right, ast.Constant):
            # self.visit_Constant(node.right)  # TODO: WHAT TO DO WITH THIS VALUE AND IS IT USEFUL?
            pass
        elif isinstance(node.right, ast.Subscript):
            # self.visit_Subscript(node.right)  # TODO: WHAT TO DO WITH THIS VALUE AND IS IT USEFUL?
            pass
        elif isinstance(node.right, ast.Compare):
            self.visit_Compare(node.right)
        else:
            print("RIGHT VALUE:", node.__dict__)
            print(astor.to_source(node))

        left_package = get_package_info(left, self.alias)
        right_package = get_package_info(right, self.alias)
        if left_package is not None and right_package is not None:
            if left_package == pd_dataframe or right_package == pd_dataframe:
                return f'{pd_dataframe.library_path}.{pd_dataframe.name}'
            return astor.to_source(node)  # TODO: FIX THIS RETURN TYPE
        elif left_package is not None and len(left_package.return_types) > 0:
            pack = left_package.return_types[0]
            return f"{pack.library_path}.{pack.name}"
        elif right_package is not None and len(right_package.return_types) > 0:
            pack = right_package.return_types[0]
            return f"{pack.library_path}.{pack.name}"
        else:
            return astor.to_source(node)

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        pass

    def visit_Lambda(self, node: Lambda) -> Any:
        pass

    def visit_IfExp(self, node: IfExp) -> Any:
        pass

    def visit_Dict(self, node: Dict) -> Any:
        keys = []
        values = []

        for key in node.keys:
            if isinstance(key, ast.Constant):
                keys.append(self.visit_Constant(key))
            else:
                print("DICT KEY:", node)
                print(astor.to_source(node))

        for value in node.values:
            if isinstance(value, ast.Constant):
                values.append(self.visit_Constant(value))
            elif isinstance(value, ast.Name):
                values.append(self.visit_Name(value))
            elif isinstance(value, ast.Attribute):
                attr_value, _ = self.visit_Attribute(value)
                values.append(attr_value)
            # elif isinstance(value, ast.List):
            #     print(self.visit_List(value))  # TODO: TO FIX
            elif isinstance(value, ast.List):
                values.append(self.visit_List(value))
            else:
                print("DICT VALUE:", node.values)
                print(astor.to_source(node))

        return {a: b for a, b in tuple(zip(keys, values))}

    def visit_Set(self, node: Set) -> Any:
        pass

    def visit_ListComp(self, node: ListComp) -> Any:
        pass

    def visit_SetComp(self, node: SetComp) -> Any:
        pass

    def visit_DictComp(self, node: DictComp) -> Any:
        pass

    def visit_GeneratorExp(self, node: GeneratorExp) -> Any:
        pass

    def visit_Await(self, node: Await) -> Any:
        pass

    def visit_Yield(self, node: Yield) -> Any:
        pass

    def visit_YieldFrom(self, node: YieldFrom) -> Any:
        pass

    def visit_Compare(self, node: Compare) -> Any:
        if isinstance(node.left, ast.Subscript):
            self.visit_Subscript(node.left)
        else:
            print("COMPARE LEFT:", node.__dict__)
            print(astor.to_source(node))

    def _extract_arguments(self, path_to_library: str, arguments: list, keywords: list):
        return packages.get(path_to_library).parameters

    def visit_Call(self, node: Call) -> Any:
        package = file = base_package = None
        if isinstance(node.func, ast.Name):
            package = self.visit_Name(node.func)
        elif isinstance(node.func, ast.Attribute):
            package, base_package = self.visit_Attribute(node.func)
            self._extract_dataflow(base_package)
            f = self.files.get(base_package)
            if f is not None and len(self.columns) > 0:
                self.graph_info.add_columns(f.filename, self.columns)
                self.columns.clear()
        else:
            print("CALL FUNC:", node.__dict__)
            print(astor.to_source(node))

        if package is not None and not isinstance(package, list):  # TODO: Extract this as method
            parent_library, *rest = package.split('.')
            pkg = self.variables.get(parent_library)
            if pkg is not None and type(pkg) not in (str, list, int):
                base_package = parent_library
                package = f"{pkg.library_path}.{pkg.name}.{'.'.join(rest)}"

            linked_file = self.files.get(parent_library)
            if linked_file is not None and len(self.columns) > 0:
                self.graph_info.add_columns(linked_file.filename, self.columns)
                self.columns.clear()

        call_args = {}
        file_args = {}
        class_args = []
        is_block = False
        label = ''

        package_class = self._get_package_info(package)
        keys = iter(package_class.parameters.keys())
        parameters = package_class.parameters.copy()

        for i in range(len(node.args)):
            parameter_value = None
            if not is_block:
                label = next(keys, '')
                is_block = '*' in label
                if is_block:
                    parameters[label] = []
            if isinstance(node.args[i], ast.Constant):
                parameter_value = self.visit_Constant(node.args[i])
                self._add_to_column(parameter_value, base_package)
                file = self._file_creation(parameter_value)
                if parameter_value in self.files.keys():
                    file = self.files.get(parameter_value)
                    file_args[i] = parameter_value
                if parameter_value in self.variables.keys():
                    call_args[i] = parameter_value
                if package in self.user_defined_class:
                    class_args.append(parameter_value)
            elif isinstance(node.args[i], ast.Name):
                parameter_value = self.visit_Name(node.args[i])
                self._extract_dataflow(parameter_value)
                self._add_to_column(parameter_value, base_package)

                if parameter_value in self.files.keys():
                    file = self.files.get(parameter_value)
                    file_args[i] = parameter_value
                if parameter_value in self.variables.keys():
                    call_args[i] = parameter_value
                if parameter_value in self.variables.keys():
                    variable = self.variables.get(parameter_value)
                    if isinstance(variable, list):
                        for col_value in variable:
                            self._add_to_column(col_value, base_package)
                            file = self._file_creation(col_value)
                            if not isinstance(col_value, list) and col_value in self.files.keys():
                                file = self.files.get(col_value)
                                file_args[i] = col_value
            elif isinstance(node.args[i], ast.Call):
                psg = self._param_subgraph_init()
                psg.subgraph_node = self.subgraph_node
                psg.subgraph = self.subgraph
                psg.visit(node.args[i])
                if psg.return_type is not None:
                    if len(psg.return_type) == 1:
                        parameter_value = psg.return_type[0].name
                    else:
                        parameter_value = [psg.return_type[i].name for i in range(len(psg.return_type))]
            elif isinstance(node.args[i], ast.List):
                parameter_value = self.visit_List(node.args[i])
                for arg_list in parameter_value:
                    self._add_to_column(arg_list, base_package)
                    if isinstance(arg_list, str) and arg_list in self.files.keys():
                        file = self.files.get(arg_list)
            elif isinstance(node.args[i], ast.Dict):
                parameter_value = self.visit_Dict(node.args[i])
            elif isinstance(node.args[i], ast.Subscript):
                arg_subscript = self.visit_Subscript(node.args[i])
                if not isinstance(arg_subscript, list):
                    arg_package = self.variables.get(arg_subscript, None)
                    if isinstance(arg_package, Calls.Call):
                        parameter_value = arg_package.name
                    elif arg_package is not None:
                        parameter_value = arg_package
            elif isinstance(node.args[i], ast.Lambda):
                parameter_value = _format_node_text(node.args[i])
            else:
                print("CALL ARG:", node.__dict__)
                print(astor.to_source(node))

            if package_class.is_relevant:
                _insert_parameter(parameters, is_block, label, parameter_value)

        for kw in node.keywords:
            if isinstance(kw, ast.keyword):
                edge, value = self.visit_keyword(kw)
                self._extract_dataflow(value)
                self._add_to_column(value, base_package)
                if package_class.is_relevant:
                    parameters[edge] = str(value)

        self.graph_info.add_parameters(parameters)
        self._create_package_call(package_class, package)

        if base_package is not None:
            variable, *_ = base_package.split('.')
            current_file = self.files.get(variable)
            if current_file is not None and len(self.columns) > 0:
                self.graph_info.add_columns(current_file.filename, self.columns)
                self.columns = []

        if package in self.user_defined_class:
            return_type = self._class_subgraph_logic(package, class_args)
        elif type(package) != list and package in self.subgraph.keys():
            return_type = self._subgraph_logic(package, file_args, call_args)
            return return_type, file, package, base_package
        elif package_class.is_relevant:
            return package_class.return_types, file, package, base_package
        return [], file, package, base_package

    def visit_FormattedValue(self, node: FormattedValue) -> Any:
        pass

    def visit_JoinedStr(self, node: JoinedStr) -> Any:
        pass

    def visit_Constant(self, node: Constant) -> Any:
        return node.value

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        pass

    def visit_Attribute(self, node: Attribute) -> Any:
        if isinstance(node.value, ast.Name):
            value = self.visit_Name(node.value)
            is_column = self._add_to_column(node.attr, value)
            package = self.variables.get(value)
            if isinstance(package, str):
                return f"{package}{'' if is_column else f'.{node.attr}'}", value
            elif isinstance(package, list):
                return [f"{el}{'' if is_column else f'.{node.attr}'}" for el in package], value
            elif type(package) in (int, float):
                return None, value
            elif package is not None:
                return f"{package.library_path}.{package.name}{'' if is_column else f'.{node.attr}'}", value
            return f"{value}.{node.attr}", value
        elif isinstance(node.value, ast.Subscript):
            value = self.visit_Subscript(node.value)
            return f'{value}.{node.attr}', None
        elif isinstance(node.value, ast.Call):
            subgraph = self._param_subgraph_init()
            subgraph.target_node = self.graph_info.tail if self.target_node is None else self.target_node
            return_types, file, _, base = subgraph.visit(node.value)
            self.graph_info.add_concurrent_flow(self.target_node)
            if len(return_types) > 0:
                return f"{return_types[0].library_path}.{return_types[0].name}.{node.attr}", base
            else:
                print("ATTRIBUTE VALUE CALL: NO RETURN TYPE")
        elif isinstance(node.value, ast.BinOp):
            values = self.visit_BinOp(node.value)
            return f"{values}.{node.attr}", None
        elif isinstance(node.value, ast.Attribute):
            values, base = self.visit_Attribute(node.value)
            return f"{values}.{node.attr}", base
        else:
            print("ATTRIBUTE VALUE:", node.__dict__)
            print(astor.to_source(node))
        return None, None

    def visit_Subscript(self, node: Subscript) -> Any:
        name = ''
        if isinstance(node.value, ast.Name):
            name = self.visit_Name(node.value)
        elif isinstance(node.value, ast.Call):
            return_types, file, name, _ = self.visit_Call(node.value)
            if name is not None and isinstance(name, str):  # TODO: VERIFY THIS
                name = name.split('.')[0]
        elif isinstance(node.value, ast.Attribute):
            name, _ = self.visit_Attribute(node.value)
            if name is not None and isinstance(name, str):
                name = name.split('.')[0]
        elif isinstance(node.value, ast.Subscript):
            name = self.visit_Subscript(node.value)
        else:
            print("SUBSCRIPT VALUE", node.__dict__)
            print(astor.to_source(node))

        if not isinstance(name, list):
            var = self.variables.get(name)
            if var == pd_dataframe and name in self.files.keys():
                if isinstance(node.slice, ast.Index):
                    index = self.visit_Index(node.slice)
                    file_name = self.files.get(name).filename
                    file = self.working_file.get(file_name, pd.DataFrame())
                    column_list = list(file)

                    if not isinstance(index, list):
                        index = self.variables.get(index, index)
                    if isinstance(index, int):
                        if index < len(column_list):
                            column = column_list[index]
                            self.columns.append(column)
                    elif isinstance(index, str) and index in column_list:
                        self.columns.append(index)
                    elif isinstance(index, list):
                        for value in index:
                            if value in column_list:
                                self.columns.append(value)
                elif isinstance(node.slice, ast.ExtSlice):
                    self.visit_ExtSlice(node.slice)  # TODO: HOW TO EXPRESS THIS VALUE?
                else:
                    print('SUBSCRIPT SOMETHING', node.slice, node.__dict__)
                    print(astor.to_source(node))
        return name

    def visit_Starred(self, node: Starred) -> Any:
        pass

    def visit_Name(self, node: Name) -> Any:
        return node.id

    def visit_List(self, node: List) -> Any:
        elements = []
        for el in node.elts:
            if isinstance(el, ast.Constant):
                elements.append(self.visit_Constant(el))
            elif isinstance(el, ast.Name):
                elements.append(self.visit_Name(el))
            elif isinstance(el, ast.Call):
                package, file, name, _ = self.visit_Call(el)
                elements.append(name)  # TODO: VERIFY HOW TO RETURN THE VALUE
            elif isinstance(el, ast.List):
                elements.append(self.visit_List(el))
            elif isinstance(el, ast.Subscript):
                elements.append(self.visit_Subscript(el))
            else:
                print("LIST VALUE:", node.__dict__)
                print(astor.to_source(node))
        return elements

    def visit_Tuple(self, node: Tuple) -> Any:
        elements = []
        for element in node.elts:
            if isinstance(element, ast.Name):
                elements.append(self.visit_Name(element))
            elif isinstance(element, ast.Constant):
                elements.append(self.visit_Constant(element))
            elif isinstance(element, ast.Subscript):
                elements.append(self.visit_Subscript(element))
            else:
                print("TUPLE ELTS:", node.__dict__)
                print(astor.to_source(node))
        return elements

    def visit_Del(self, node: Del) -> Any:
        pass

    def visit_Load(self, node: Load) -> Any:
        pass

    def visit_Store(self, node: Store) -> Any:
        pass

    def visit_And(self, node: And) -> Any:
        pass

    def visit_Or(self, node: Or) -> Any:
        pass

    def visit_Add(self, node: Add) -> Any:
        pass

    def visit_BitAnd(self, node: BitAnd) -> Any:
        pass

    def visit_BitOr(self, node: BitOr) -> Any:
        pass

    def visit_BitXor(self, node: BitXor) -> Any:
        pass

    def visit_Div(self, node: Div) -> Any:
        pass

    def visit_FloorDiv(self, node: FloorDiv) -> Any:
        pass

    def visit_LShift(self, node: LShift) -> Any:
        pass

    def visit_Mod(self, node: Mod) -> Any:
        pass

    def visit_Mult(self, node: Mult) -> Any:
        pass

    def visit_MatMult(self, node: MatMult) -> Any:
        pass

    def visit_Pow(self, node: Pow) -> Any:
        pass

    def visit_RShift(self, node: RShift) -> Any:
        pass

    def visit_Sub(self, node: Sub) -> Any:
        pass

    def visit_Invert(self, node: Invert) -> Any:
        pass

    def visit_Not(self, node: Not) -> Any:
        pass

    def visit_UAdd(self, node: UAdd) -> Any:
        pass

    def visit_USub(self, node: USub) -> Any:
        pass

    def visit_Eq(self, node: Eq) -> Any:
        pass

    def visit_Gt(self, node: Gt) -> Any:
        pass

    def visit_GtE(self, node: GtE) -> Any:
        pass

    def visit_In(self, node: In) -> Any:
        pass

    def visit_Is(self, node: Is) -> Any:
        pass

    def visit_IsNot(self, node: IsNot) -> Any:
        pass

    def visit_Lt(self, node: Lt) -> Any:
        pass

    def visit_LtE(self, node: LtE) -> Any:
        pass

    def visit_NotEq(self, node: NotEq) -> Any:
        pass

    def visit_NotIn(self, node: NotIn) -> Any:
        pass

    def visit_comprehension(self, node: comprehension) -> Any:
        pass

    def visit_ExceptHandler(self, node: ExceptHandler) -> Any:
        pass

    def visit_arguments(self, node: arguments) -> Any:
        arguments_list = []
        for argument in node.args:
            arguments_list.append(self.visit_arg(argument))
        return arguments_list

    def visit_arg(self, node: arg) -> Any:
        return node.arg

    def visit_keyword(self, node: keyword) -> Any:
        value = None
        if isinstance(node.value, ast.Constant):
            value = self.visit_Constant(node.value)
        elif isinstance(node.value, ast.Name):
            value = self.visit_Name(node.value)
        elif isinstance(node.value, ast.List):
            value = self.visit_List(node.value)
        elif isinstance(node.value, ast.Tuple):
            value = self.visit_Tuple(node.value)
        elif isinstance(node.value, ast.BinOp):
            value = self.visit_BinOp(node.value)
        elif isinstance(node.value, ast.Dict):
            value = self.visit_Dict(node.value)
        elif isinstance(node.value, ast.Subscript):
            value = self.visit_Subscript(node.value)
        elif isinstance(node.value, ast.Attribute):
            value, _ = self.visit_Attribute(node.value)
        else:
            print("KEYWORD VALUE:", node.__dict__)
        return node.arg, value

    def visit_alias(self, node: alias) -> Any:
        return node.name, node.asname

    def visit_withitem(self, node: withitem) -> Any:
        pass

    def visit_ExtSlice(self, node: ExtSlice) -> Any:
        for dim in node.dims:
            sl = None
            if isinstance(dim, ast.Slice):
                sl = self.visit_Slice(dim)
            elif isinstance(dim, ast.Index):
                sl = self.visit_Index(dim)
            else:
                print("EXT_SLICE DIM:", node.__dict__)
                print(astor.to_source(node))
        pass

    def visit_Index(self, node: Index) -> Any:
        if isinstance(node.value, ast.Constant):
            return self.visit_Constant(node.value)
        elif isinstance(node.value, ast.BinOp):
            self.visit_BinOp(node.value)
        elif isinstance(node.value, ast.Compare):
            self.visit_Compare(node.value)
        elif isinstance(node.value, ast.List):
            return self.visit_List(node.value)
        elif isinstance(node.value, ast.Subscript):
            print(self.visit_Subscript(node.value))  # TODO: MAKE SOMETHING OF THIS VALUE
        elif isinstance(node.value, ast.Name):
            return self.visit_Name(node.value)
        else:
            print("INDEX VALUE:", node.__dict__)
            print(astor.to_source(node))

    def visit_Suite(self, node: Suite) -> Any:
        pass

    def visit_AugLoad(self, node: AugLoad) -> Any:
        pass

    def visit_AugStore(self, node: AugStore) -> Any:
        pass

    def visit_Param(self, node: Param) -> Any:
        pass

    def visit_Num(self, node: Num) -> Any:
        pass

    def visit_Str(self, node: Str) -> Any:
        pass

    def visit_Bytes(self, node: Bytes) -> Any:
        pass

    def visit_NameConstant(self, node: NameConstant) -> Any:
        pass

    def visit_Ellipsis(self, node: Ellipsis) -> Any:
        pass

    def _add_to_column(self, column_name, table_name=None):
        file = self.files.get(table_name, File(''))
        file_name = file.filename
        working_file = self.working_file.get(file_name, pd.DataFrame())
        if column_name in list(working_file):
            self.columns.append(column_name)
            return True
        return False

    def _subgraph_logic(self, package, file_args, call_args):
        s_graph = self.subgraph.get(package)
        s_graph.data_flow_container = self.data_flow_container.copy()

        for param in s_graph.arguments.values():
            s_graph.data_flow_container[param] = self.graph_info.tail

        s_graph.files = {s_graph.arguments.get(file_k): self.files.get(file_args.get(file_k))
                         for file_k in file_args.keys()}
        s_graph.variables = {s_graph.arguments.get(arg_k): self.variables.get(call_args.get(arg_k))
                             for arg_k in call_args.keys()}
        s_graph.is_starting = True
        s_graph.visit(self.subgraph_node.get(package))
        return s_graph.return_type

    def _class_subgraph_logic(self, package, class_args):
        class_graph = self.subgraph.get(package)

    def _param_subgraph_init(self):
        psg = ParamSubGraph()
        psg.graph_info = self.graph_info
        psg.working_file = self.working_file
        psg.files = self.files
        psg.variables = self.variables
        psg.packages = self.packages
        psg.alias = self.alias
        psg.data_flow_container = self.data_flow_container
        return psg

    def _create_package_call(self, package_class, package):
        if package_class.is_relevant:
            self.graph_info.add_package_call(package_class.full_path())
        elif isinstance(package, str) and package in ("len", "range", "list"):
            self.graph_info.add_built_in_call(package)

    def _file_creation(self, file: str) -> File:
        if is_file(file):
            _, filename = os.path.split(file)
            file = File(filename)
            self.graph_info.add_file(file)
            return file

    def _extract_dataflow(self, variable_name):
        if not isinstance(variable_name, str):
            return
        flow = self.data_flow_container.get(variable_name)
        if flow is not None:
            self.graph_info.add_data_flows(flow)

    def _create_library_path(self, package_name):
        if not package_name or isinstance(package_name, list):
            return ''
        if '.' in package_name:
            base, *rest = package_name.split('.')
            base = self.alias.get(base, base)
            return f'{base}.{".".join(rest)}'
        return self.library_path.get(package_name, package_name)

    def _get_package_info(self, package_name):
        library_path = self._create_library_path(package_name)
        return packages.get(library_path, Calls.Call(is_relevant=False))


class SubGraph(NodeVisitor):
    __slots__ = ['arguments', 'is_starting', 'return_type']

    def __init__(self):
        super().__init__()
        self.is_starting = True
        self.return_type = None

    def visit(self, node: AST) -> Any:
        if self.is_starting:
            self.is_starting = False
            return self.visit_FunctionDef(node)
        return super().visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        self.control_flow.append(ControlFlow.METHOD)
        for el in node.body:
            self.visit(el)
        self.control_flow.pop()

    def visit_Return(self, node: Return) -> Any:
        if isinstance(node.value, ast.Call):
            self.visit_Call(node.value)
        elif isinstance(node.value, ast.Name):
            name = self.visit_Name(node.value)
            var = self.variables.get(name)
            if var is not None and isinstance(var, Calls.Call):
                self.return_type = var.return_types
        else:
            print("SUBGRAPH RETURN VALUE:", node.__dict__)
            print(astor.to_source(node))


class ParamSubGraph(NodeVisitor):
    __slots__ = ['return_type', 'target_node']

    def __init__(self):
        super().__init__()
        self.return_type = None

    def visit(self, node: AST) -> Any:
        n = super().visit(node)
        if self.target_node is None:
            self.graph_info.rewrite_node_flow()
        else:
            self.graph_info.insert_before(self.target_node)
        return n

    def visit_Call(self, node: Call) -> Any:  # TODO: REMOVE UNUSED ELEMENT
        return_types, file, name, base = super().visit_Call(node)
        info = None
        # if package is not None:
        #     *path, lib = package.split('.')
        #     info = get_package_info(lib, '.'.join(path))
        if info is not None:
            self.return_type = return_types[0] if len(return_types) > 0 else None
        self.return_type = return_types
        return return_types, file, name, base


class ForSubGraph(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.is_starting = True

    def visit(self, node: AST) -> Any:
        if self.is_starting:
            return self.visit_For(node)
        return super().visit(node)

    def visit_For(self, node: For) -> Any:
        if self.is_starting:
            self.is_starting = False
            target = iter_value = None
            if isinstance(node.target, ast.Name):
                target = self.visit_Name(node.target)
            else:
                print("FOR TARGET", node.__dict__)
                print(astor.to_source(node))

            if isinstance(node.iter, ast.Call):
                self.visit_Call(node.iter)
            elif isinstance(node.iter, ast.Name):
                iter_value = self.visit_Name(node.iter)
            elif isinstance(node.iter, ast.List):
                self.variables[target] = self.visit_List(node.iter)
            else:
                print("FOR ITER:", node.__dict__)
                print(astor.to_source(node))

            if iter_value in self.variables.keys():
                self.variables[target] = self.variables.get(iter_value)

            for el in node.body:
                self.visit(el)
        else:
            super().visit_For(node)


class ClassSubGraph(NodeVisitor):
    __slots__ = ['arguments', 'return_type']

    def __init__(self):
        super().__init__()
        self.is_starting = True
        self.return_type = None

    def visit(self, node: AST) -> Any:
        if self.is_starting:
            self.is_starting = False
            return self.visit_FunctionDef(node)
        return super().visit(node)

    def visit_init(self, node: FunctionDef):
        args = [function_arg for function_arg in self.visit_arguments(node.args) if arg != 'self']
        # for function_arg in args:
        #     if
        # for el in node.body:
        #     if isinstance(el, ast.Assign):
        #
        #     else:
        #         print("CLASS INIT BODY", node.__dict__)
        #         print(astor.to_source(node))
        print(node.__dict__)
        pass

    def visit_method(self, node: FunctionDef):
        self.control_flow.append(ControlFlow.METHOD)
        print(node.__dict__)
        self.control_flow.pop()
        pass

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        if node is None:
            return self.visit_init(node)
        for el in node.body:
            self.visit(el)
        self.control_flow.pop()

    def visit_Return(self, node: Return) -> Any:
        if isinstance(node.value, ast.Call):
            self.visit_Call(node.value)
        elif isinstance(node.value, ast.Name):
            name = self.visit_Name(node.value)
            self.return_type = self.variables.get(name).return_types
            print(self.variables.get(name))
        else:
            print("SUBGRAPH RETURN VALUE:", node.__dict__)
            print(astor.to_source(node))
