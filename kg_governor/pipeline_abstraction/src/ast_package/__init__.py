import ast
import astor
from typing import cast

import src.Calls as Calls
from src.util import format_node_text
from src.ast_package.types import CallComponents, CallArgumentsComponents, AssignComponents, BinOpComponents, \
    AttributeComponents


class AstPackage:
    def extract_func(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallComponents):
        print("CALL FUNC:", node.__dict__)
        print(astor.to_source(node))

    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        print("CALL ARG:", node.__dict__)
        print(astor.to_source(node))

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        print("ASSIGN VALUE:", node.__dict__)
        print(astor.to_source(node))

    def analyze_assign_target(self, node_visitor: ast.NodeVisitor, node: ast, components: AssignComponents):
        print("ASSIGN TARGET:", node.__dict__)
        print(astor.to_source(node))

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        print("KEYWORD VALUE:", node.__dict__)

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int, list_elements: list):
        print("LIST VALUE:", node.__dict__)
        print(astor.to_source(node))

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        print(f"BinOp {side.upper()} BRANCH VALUE:", node.__dict__)
        print(astor.to_source(node))

    def extract_subscript_value(self, node_visitor: ast.NodeVisitor, node: ast.Subscript):
        print("SUBSCRIPT VALUE", node.__dict__)
        print(astor.to_source(node))
        return None, None

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        print("ATTRIBUTE VALUE:", node.__dict__)
        print(astor.to_source(node))


class Name(AstPackage):
    def extract_func(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallComponents):
        components.package = node_visitor.visit_Name(cast(ast.Name, node.func))

    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        parameter_value = node_visitor.visit_Name(cast(ast.Name, node.args[pos]))
        node_visitor._extract_dataflow(parameter_value)
        node_visitor._add_to_column(parameter_value, call_components.base_package)

        if parameter_value in node_visitor.files.keys():
            call_components.file = node_visitor.files.get(parameter_value)
            components.file_args[pos] = parameter_value
        if parameter_value in node_visitor.variables.keys():
            components.call_args[pos] = parameter_value
            variable = node_visitor.variables.get(parameter_value)
            if isinstance(variable, list):
                for col_value in variable:
                    node_visitor._add_to_column(col_value, call_components.base_package)
                    call_components.file = node_visitor._file_creation(col_value)
                    if not isinstance(col_value, list) and col_value in node_visitor.files.keys():
                        call_components.file = node_visitor.files.get(col_value)
                        components.file_args[pos] = col_value

        return parameter_value

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_Name(cast(ast.Name, node.value))

    def analyze_assign_target(self, node_visitor: ast.NodeVisitor, node: ast, components: AssignComponents):
        name = node_visitor.visit_Name(node)
        node_visitor.data_flow_container[name] = node_visitor.graph_info.tail
        if type(components.value) == str:
            if components.file is not None:
                node_visitor.files[name] = components.file
            elif components.value in node_visitor.files.keys():
                node_visitor.files[name] = node_visitor.files.get(components.value)
            if components.value in node_visitor.variables.keys():
                node_visitor.variables[name] = node_visitor.variables.get(components.value)
        elif isinstance(components.value, list):
            if len(components.value) > 0:
                node_visitor.variables[name] = components.value[0]
            if components.file is not None:
                node_visitor.files[name] = components.file

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_Name(node.value)

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int,  list_elements: list):
        list_elements.append(node_visitor.visit_Name(cast(ast.Name, node.elts[pos])))

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        name = node_visitor.visit_Name(getattr(node, side))
        node_visitor._extract_dataflow(name)

    def extract_subscript_value(self, node_visitor: ast.NodeVisitor, node: ast.Subscript):
        return node_visitor.visit_Name(cast(ast.Name, node.value)), None

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        value = node_visitor.visit_Name(cast(ast.Name, node.value))
        is_column = node_visitor._add_to_column(node.attr, value)
        package = node_visitor.variables.get(value)  # TODO: Makes variables all packages

        if isinstance(package, str):
            components.path = f"{package}{'' if is_column else f'.{node.attr}'}"
            components.parent_path = value
        elif isinstance(package, list):
            components.path = [f"{el}{'' if is_column else f'.{node.attr}'}" for el in package]
            components.parent_path = value
        elif type(package) in (int, float):
            components.parent_path = value
        elif package is not None:
            components.path = f"{package.library_path}.{package.name}{'' if is_column else f'.{node.attr}'}"
            components.parent_path = value
        else:
            components.path = f"{value}.{node.attr}"
            components.parent_path = value


class Attribute(AstPackage):
    def extract_func(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallComponents):
        components.package, components.base_package, components.file = node_visitor.visit_Attribute(cast(ast.Attribute, node.func))

        node_visitor._extract_dataflow(components.base_package)
        f = node_visitor.files.get(components.base_package)
        node_visitor._connect_node_to_column(f)

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value, base, components.file = node_visitor.visit_Attribute(cast(ast.Attribute, node.value))
        node_visitor._connect_node_to_column(node_visitor.files.get(base))

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        value, _, _ = node_visitor.visit_Attribute(node.value)
        return value

    def extract_subscript_value(self, node_visitor: ast.NodeVisitor, node: ast.Subscript):
        name, base, _ = node_visitor.visit_Attribute(cast(ast.Attribute, node.value))
        try:
            if name is not None and isinstance(name, str):
                package = Calls.packages.get(name)
                name = package.return_types[0].full_path()
        finally:
            return name, base

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        path, components.parent_path, components.file = node_visitor.visit_Attribute(cast(ast.Attribute, node.value))

        components.path = f"{path}.{node.attr}"


class Constant(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        parameter_value = node_visitor.visit_Constant(cast(ast.Constant, node.args[pos]))

        if call_components.file:
            return parameter_value

        node_visitor._add_to_column(parameter_value, call_components.base_package)
        call_components.file = node_visitor._file_creation(parameter_value)

        if parameter_value in node_visitor.files.keys():
            call_components.file = node_visitor.files.get(parameter_value)
            components.file_args[pos] = parameter_value
        if parameter_value in node_visitor.variables.keys():
            components.call_args[pos] = parameter_value
        if call_components.package in node_visitor.user_defined_class:
            components.class_args.append(parameter_value)

        return parameter_value

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_Constant(cast(ast.Constant, node.value))
        components.file = node_visitor._file_creation(components.value)

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_Constant(node.value)

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int,  list_elements: list):
        list_elements.append(node_visitor.visit_Constant(cast(ast.Constant, node.elts[pos])))

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        node_visitor.visit_Constant(getattr(node, side))
        setattr(components, side, None)


class Call(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        psg = node_visitor.param_subgraph_init()
        psg.subgraph_node = node_visitor.subgraph_node
        psg.subgraph = node_visitor.subgraph
        psg.visit(node.args[pos])

        if psg.return_type is None:
            return None
        if len(psg.return_type) == 1:
            return psg.return_type[0].name
        return [psg.return_type[i].name for i in range(len(psg.return_type))]

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value, components.file, _, _ = node_visitor.visit_Call(cast(ast.Call, node.value))

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        _, _, _, base = node_visitor.visit_Call(cast(ast.Call, node.value))
        return base

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int,  list_elements: list):
        package, file, name, _ = node_visitor.visit_Call(cast(ast.Call, node.elts[pos]))
        list_elements.append(name)  # TODO: VERIFY HOW TO RETURN THE VALUE

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        subgraph = node_visitor.param_subgraph_init()
        subgraph.target_node = node_visitor.graph_info.tail if node_visitor.target_node is None else node_visitor.target_node
        value, _, _, _ = subgraph.visit(getattr(node, side))
        setattr(components, side, value)
        if value is not None and len(value) > 0:
            setattr(components, side, value[0])

    def extract_subscript_value(self, node_visitor: ast.NodeVisitor, node: ast.Subscript):
        return_types, file, name, base = node_visitor.visit_Call(cast(ast.Call, node.value))
        try:
            if name is not None and isinstance(name, str):
                package = Calls.packages.get(name)  # TODO: Verify that this is working
                name = package.return_types[0].full_path(), base
        finally:
            return name, base

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        subgraph = node_visitor.param_subgraph_init()
        subgraph.target_node = node_visitor.graph_info.tail if node_visitor.target_node is None else node_visitor.target_node
        return_types, file, _, base = subgraph.visit(node.value)
        node_visitor.graph_info.add_concurrent_flow(node_visitor.target_node)

        is_column = node_visitor._add_to_column(node.attr, base)

        if len(return_types) > 0:
            components.path = f"{return_types[0].library_path}.{return_types[0].name}{'' if is_column else f'.{node.attr}'}"
            components.parent_path = base
            components.file = file
        else:
            print("ATTRIBUTE VALUE CALL: NO RETURN TYPE")


class List(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        parameter_value = node_visitor.visit_List(cast(ast.List, node.args[pos]))
        for arg_list in parameter_value:
            node_visitor._add_to_column(arg_list, call_components.base_package)
            if isinstance(arg_list, str) and arg_list in node_visitor.files.keys():
                call_components.file = node_visitor.files.get(arg_list)

        return parameter_value

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_List(cast(ast.List, node.value))
        for element in components.value:
            node_visitor._extract_dataflow(element)

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_List(node.value)

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int,  list_elements: list):
        list_elements.append(node_visitor.visit_List(cast(ast.List, node.elts[pos])))


class Dict(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        return node_visitor.visit_Dict(cast(ast.Dict, node.args[pos]))

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_Dict(cast(ast.Dict, node.value))

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_Dict(node.value)


class Subscript(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        arg_subscript, _ = node_visitor.visit_Subscript(cast(ast.Subscript, node.args[pos]))
        if isinstance(arg_subscript, list):
            return None

        arg_package = node_visitor.variables.get(arg_subscript, None)

        if isinstance(arg_package, Calls.Call):
            return arg_package.name
        return arg_package

    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value, _ = node_visitor.visit_Subscript(cast(ast.Subscript, node.value))
        node_visitor._extract_dataflow(components.value)

        if isinstance(components.value, list):
            return

        components.variable = node_visitor.variables.get(components.value, None)
        if components.value in node_visitor.files.keys():
            components.file = node_visitor.files.get(components.value)
            node_visitor._connect_node_to_column(components.file)

    def analyze_assign_target(self, node_visitor: ast.NodeVisitor, node: ast, components: AssignComponents):
        variable, _ = node_visitor.visit_Subscript(node)
        node_visitor._extract_dataflow(variable)
        if isinstance(variable, str):
            node_visitor.data_flow_container[variable] = node_visitor.graph_info.tail
            file = node_visitor.files.get(variable)
            node_visitor._connect_node_to_column(file)

        return variable

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        name, _ = node_visitor.visit_Subscript(node.value)
        return name

    def extract_list_element(self, node_visitor: ast.NodeVisitor, node: ast.List, pos: int,  list_elements: list):
        name, _ = node_visitor.visit_Subscript(cast(ast.Subscript, node.elts[pos]))
        list_elements.append(name)

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        element_name, _ = node_visitor.visit_Subscript(getattr(node, side))
        element_package = node_visitor.variables.get(element_name, element_name)
        setattr(components, side, element_package)

    def extract_subscript_value(self, node_visitor: ast.NodeVisitor, node: ast.Subscript):
        return node_visitor.visit_Subscript(cast(ast.Subscript, node.value))

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        value, _ = node_visitor.visit_Subscript(cast(ast.Subscript, node.value))
        components.path = f'{value}.{node.attr}'


class Lambda(AstPackage):
    def analyze_call_arguments(self, node_visitor: ast.NodeVisitor, node: ast.Call, components: CallArgumentsComponents,
                               call_components: CallComponents, pos: int):
        return format_node_text(node.args[pos])


class BinOp(AstPackage):
    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_BinOp(cast(ast.BinOp, node.value))

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_BinOp(node.value)

    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        setattr(components, side, node_visitor.visit_BinOp(getattr(node, side)))

    def analyze_attribute_value(self, node_visitor: ast.NodeVisitor, node: ast.Attribute,
                                components: AttributeComponents):
        values = node_visitor.visit_BinOp(cast(ast.BinOp, node.value))
        components.path = f"{values}.{node.attr}"


class Tuple(AstPackage):
    def extract_assign_value(self, node_visitor: ast.NodeVisitor, node: ast.Assign, components: AssignComponents):
        components.value = node_visitor.visit_Tuple(cast(ast.Tuple, node.value))

    def analyze_assign_target(self, node_visitor: ast.NodeVisitor, node: ast, components: AssignComponents):
        tuple_values = node_visitor.visit_Tuple(node)
        for el in tuple_values:
            node_visitor.data_flow_container[el] = node_visitor.graph_info.tail

        if components.value is not None and tuple_values is not None:
            for sub_target, package in tuple(zip(tuple_values, components.value)):
                node_visitor.variables[sub_target] = package
                if components.file is not None:
                    node_visitor.files[sub_target] = components.file

    def extract_keyword_value(self, node_visitor: ast.NodeVisitor, node: ast):
        return node_visitor.visit_Tuple(node.value)


class Compare(AstPackage):
    def analyze_bin_op_branch(self, node_visitor: ast.NodeVisitor, node: ast.BinOp, side: str, components: BinOpComponents):
        node_visitor.visit_Compare(getattr(node, side))


ast_packages = {
    ast.Name: Name(),
    ast.Attribute: Attribute(),
    ast.Constant: Constant(),
    ast.Call: Call(),
    ast.List: List(),
    ast.Dict: Dict(),
    ast.Subscript: Subscript(),
    ast.Lambda: Lambda(),
    ast.BinOp: BinOp(),
    ast.Tuple: Tuple(),
    ast.Compare: Compare(),
}


def get_ast_package(package: ast) -> AstPackage:
    return ast_packages.get(type(package), AstPackage())
