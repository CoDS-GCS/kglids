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
from typing import Any, cast
from collections import deque

import pandas as pd

import src.Calls as Calls
from src.datatypes import GraphInformation
from src.Calls import File, pd_dataframe, packages
from src.util import is_file, ControlFlow, format_node_text, get_package
from src.ast_package import AstPackage, get_ast_package
from src.ast_package.types import CallComponents, CallArgumentsComponents, AssignComponents, BinOpComponents, \
    AttributeComponents


def insert_parameter(parameters: dict, is_block: bool, parameter: str, value):
    print('<><><><>', parameter)
    if parameter is None:
        return
    if is_block:
        parameters[str(parameter)].append(value)
    else:
        parameters[str(parameter)] = value


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
                          ast.FunctionDef, ast.AugAssign, ast.Call, ast.Return, ast.Attribute]:  # TODO: Improve this
            self.columns.clear()
            self.graph_info.add_node(format_node_text(node))
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
        assign_components = AssignComponents()
        value_package = get_ast_package(node.value)
        value_package.extract_assign_value(self, node, assign_components)

        variable = None  # TODO: Choose better name
        for target in node.targets:
            target_package = get_ast_package(target)
            variable = target_package.analyze_assign_target(self, target, assign_components)

        if variable is not None and not isinstance(variable, list):
            self._connect_node_to_column(self.files.get(variable))

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
        lower = upper = step = None
        if isinstance(node.lower, ast.Constant):
            lower = self.visit_Constant(node.lower)

        return lower, upper, step

    def visit_BoolOp(self, node: BoolOp) -> Any:
        pass

    def visit_BinOp(self, node: BinOp) -> Any:
        components = BinOpComponents()

        left_ast_package = get_ast_package(node.left)
        left_ast_package.analyze_bin_op_branch(self, node, 'left', components)

        right_ast_package = get_ast_package(node.right)
        right_ast_package.analyze_bin_op_branch(self, node, 'right', components)

        left_package = get_package(components.left, self.alias)
        right_package = get_package(components.right, self.alias)

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
            return format_node_text(node)

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
                attr_value, _, _ = self.visit_Attribute(value)
                values.append(attr_value)
            # elif isinstance(value, ast.List):
            #     print(self.visit_List(value))  # TODO: TO FIX
            elif isinstance(value, ast.List):
                values.append(self.visit_List(value))
            elif isinstance(value, ast.Call):
                _, _, _, base = self.visit_Call(value)
                values.append(base)
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
        elif isinstance(node.left, ast.Call):
            self.visit_Call(node.left)
        else:
            print("COMPARE LEFT:", node.__dict__)
            print(astor.to_source(node))

        for comparator in node.comparators:
            if isinstance(comparator, ast.Call):
                self.visit_Call(comparator)

    def _connect_node_to_column(self, file):
        if file is None:
            return
        self.graph_info.add_columns(file.filename, self.columns)
        self.columns.clear()

    def _extract_parent_package_information(self, components: CallComponents):
        if components.package is None:
            return
        if isinstance(components.package, list):  # TODO: Create list extraction
            return

        components.extract_parent_library()
        pkg = self.variables.get(components.parent_library)
        components.rewrite_library_path(pkg)
        self._connect_node_to_column(self.files.get(components.parent_library))

    def visit_Call(self, node: Call) -> Any:
        func_package: AstPackage
        func_package = get_ast_package(node.func)

        call_components = CallComponents()
        func_package.extract_func(self, node, call_components)

        self._extract_parent_package_information(call_components)

        package_class = self._get_package_info(call_components.package)
        parameters = package_class.parameters.copy()
        args_components = CallArgumentsComponents(package_class.parameters.keys())

        for i in range(len(node.args)):
            if not args_components.is_block:
                args_components.next_label()
                args_components.set_is_block(parameters)

            args_package = get_ast_package(node.args[i])
            parameter_value = args_package.analyze_call_arguments(self, node, args_components, call_components, i)

            if package_class.is_relevant:
                insert_parameter(parameters, args_components.is_block, args_components.label, parameter_value)

        for kw in node.keywords:
            if isinstance(kw, ast.keyword):
                edge, value = self.visit_keyword(kw)
                self._extract_dataflow(value)
                self._add_to_column(value, call_components.base_package)
                if package_class.is_relevant and edge is not None:
                    parameters[edge] = str(value)

        self.graph_info.add_parameters(parameters)
        self._create_package_call(package_class, call_components.package)

        if call_components.base_package is not None:
            variable, *_ = call_components.base_package.split('.')
            self._connect_node_to_column(self.files.get(variable))

        # TODO: CLEAN THAT MESS
        if call_components.package in self.user_defined_class:
            return_type = self._class_subgraph_logic(call_components.package, args_components.class_args)
        elif type(call_components.package) != list and call_components.package in self.subgraph.keys():
            return_type = self._subgraph_logic(call_components.package,
                                               args_components.file_args,
                                               args_components.call_args)
            return return_type, call_components.file, call_components.package, call_components.base_package
        elif package_class.is_relevant:
            return (package_class.return_types,
                    call_components.file,
                    call_components.package,
                    call_components.base_package)
        return [], call_components.file, call_components.package, call_components.base_package

    def visit_FormattedValue(self, node: FormattedValue) -> Any:
        pass

    def visit_JoinedStr(self, node: JoinedStr) -> Any:
        pass

    def visit_Constant(self, node: Constant) -> Any:
        return node.value

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        pass

    def visit_Attribute(self, node: Attribute) -> Any:
        components = AttributeComponents()
        attribute_package = get_ast_package(node.value)
        attribute_package.analyze_attribute_value(self, node, components)
        return components.path, components.parent_path, components.file

    def visit_Subscript(self, node: Subscript) -> Any:
        value_package = get_ast_package(node.value)
        name, base = value_package.extract_subscript_value(self, node)

        if not isinstance(name, list):
            var = self.variables.get(name, self.variables.get(base))
            file = self.files.get(name, self.files.get(base))
            if var is pd_dataframe and file is not None:
                working_file = self.working_file.get(file.filename, pd.DataFrame())
                if isinstance(node.slice, ast.Index):
                    index = self.visit_Index(node.slice)
                    column_list = list(working_file)

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
                elif isinstance(node.slice, ast.Slice):
                    lower, upper, step = self.visit_Slice(node.slice)
                    columns = list(working_file)
                    if lower is None:
                        lower = 0
                    if upper is None:
                        upper = len(columns)
                    if step is None:
                        step = 1
                    if isinstance(lower, int):
                        for i in range(lower, upper, step):
                            self.columns.append(columns[i])
                    elif isinstance(lower, str):
                        is_checked = False
                        for col in columns:
                            if is_checked:
                                self.columns.append(col)
                                if col == upper:
                                    break
                            else:
                                if col == lower:
                                    self.columns.append(col)
                                    is_checked = True
                else:
                    print('SUBSCRIPT SOMETHING', node.slice, node.__dict__)
                    print(astor.to_source(node))
                self._connect_node_to_column(file)
        return name, base

    def visit_Starred(self, node: Starred) -> Any:
        pass

    def visit_Name(self, node: Name) -> Any:
        return node.id

    def visit_List(self, node: List) -> Any:
        elements = []
        for i in range(len(node.elts)):
            element_package = get_ast_package(node.elts[i])
            element_package.extract_list_element(self, node, i, elements)

        return elements

    def visit_Tuple(self, node: Tuple) -> Any:
        elements = []
        for element in node.elts:
            if isinstance(element, ast.Name):
                elements.append(self.visit_Name(element))
            elif isinstance(element, ast.Constant):
                elements.append(self.visit_Constant(element))
            elif isinstance(element, ast.Subscript):
                name, _ = self.visit_Subscript(element)
                elements.append(name)
            elif isinstance(element, ast.Call):
                _, _, _, base = self.visit_Call(element)
                elements.append(base)
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
        keyword_package = get_ast_package(node.value)
        value = keyword_package.extract_keyword_value(self, node)
        return node.arg, value

    def visit_alias(self, node: alias) -> Any:
        return node.name, node.asname

    def visit_withitem(self, node: withitem) -> Any:
        pass

    def visit_ExtSlice(self, node: ast.ExtSlice) -> Any:
        for dim in node.dims:
            sl = None
            if isinstance(dim, ast.Slice):
                lower, upper, step = self.visit_Slice(dim)
            elif isinstance(dim, ast.Index):
                sl = self.visit_Index(dim)
            else:
                print("EXT_SLICE DIM:", node.__dict__)
                print(astor.to_source(node))
        pass

    def visit_Index(self, node: ast.Index) -> Any:
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

    def visit_Suite(self, node: ast.Suite) -> Any:
        pass

    def visit_AugLoad(self, node: ast.AugLoad) -> Any:
        pass

    def visit_AugStore(self, node: ast.AugStore) -> Any:
        pass

    def visit_Param(self, node: ast.Param) -> Any:
        pass

    def visit_Num(self, node: ast.Num) -> Any:
        pass

    def visit_Str(self, node: ast.Str) -> Any:
        pass

    def visit_Bytes(self, node: ast.Bytes) -> Any:
        pass

    def visit_NameConstant(self, node: ast.NameConstant) -> Any:
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

    def param_subgraph_init(self):
        psg = ParamSubGraph()
        psg.graph_info = self.graph_info
        psg.working_file = self.working_file
        psg.files = self.files
        psg.variables = self.variables
        psg.packages = self.packages
        psg.alias = self.alias
        psg.data_flow_container = self.data_flow_container
        psg.library_path = self.library_path
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


# TODO: Extract sub graphs types

class SubGraph(NodeVisitor):
    __slots__ = ['arguments', 'is_starting', 'return_type']

    def __init__(self):
        super().__init__()
        self.is_starting = True
        self.return_type = None

    def visit(self, node: AST) -> Any:
        if self.is_starting:
            self.is_starting = False
            return self.visit_FunctionDef(cast(ast.FunctionDef, node))
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
        #     info = get_package(lib, '.'.join(path))
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
            return self.visit_For(cast(ast.For, node))
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
            return self.visit_FunctionDef(cast(ast.FunctionDef, node))
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
            return self.visit_init(cast(ast.FunctionDef, node))
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


class ClassObject:
    def __init__(self):
        self.variables = {}
        self.functions = {}

    def visit(self, node_visitor: NodeVisitor, function_name: str):
        package = self.functions.get(function_name)
        subgraph: NodeVisitor = node_visitor.param_subgraph_init()
        node_visitor.variables = self.variables  # TODO: join the node_visitor variables together
        return subgraph.visit(package)
