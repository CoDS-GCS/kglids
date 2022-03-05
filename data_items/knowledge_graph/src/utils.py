import re
import zlib

from camelsplit import camelsplit

from label import Label
import pandas as pd
from graphviz import Digraph


def generate_label(col_name: str, lan: str) -> Label:
    if '.csv' in col_name:
        col_name = re.sub('.csv', '', col_name)
    col_name = re.sub('[^0-9a-zA-Z]+', ' ', col_name)
    text = " ".join(camelsplit(col_name.strip()))
    text = re.sub('\s+', ' ', text.strip())
    return Label(text.lower(), lan)


def generate_component_id(dataset_name: str, table_name: str = '', column_name: str = ''):
    return zlib.crc32(bytes(dataset_name + table_name + column_name, 'utf-8'))


def generate_graphviz2(df: pd.DataFrame, predicate: str, hops: int):
    dot = Digraph()
    nodes = []
    dataset_id = ''
    table_id = ''
    column_id = ''
    keys = df.columns
    group = 0  # to color the stating point and target
    for i in range(len(df)):
        row = df.iloc[i]
        dataset_name = ''
        table_name = ''
        counter = 0
        similar = []
        for k in keys:
            if 'path' in k:
                continue

            if counter == 0:  # if dataset
                dataset_name = row[k]
                dataset_id = str(generate_component_id(dataset_name))
                if dataset_id not in nodes:
                    nodes.append(dataset_id)
                    if group == 0:
                        dot.node(dataset_id, dataset_name, style='filled', fillcolor='lightblue2')
                    elif group == hops:
                        dot.node(dataset_id, dataset_name, style='filled', fillcolor='coral')
                    else:
                        dot.node(dataset_id, dataset_name)
            if counter == 1:  # if table
                table_name = row[k]
                table_id = str(generate_component_id(dataset_name, table_name))
                if table_id not in nodes:
                    nodes.append(table_id)
                    if group == 0:
                        dot.node(table_id, table_name, style='filled', fillcolor='lightblue2')
                    elif group == hops:
                        dot.node(table_id, table_name, style='filled', fillcolor='coral')
                    else:
                        dot.node(table_id, table_name)
            if counter == 2:  # if columns
                column_name = row[k]
                column_id = str(generate_component_id(dataset_name, table_name, column_name))
                similar.append(column_id)
                if column_id not in nodes:
                    if group == 0:
                        dot.node(column_id, column_name, style='filled', fillcolor='lightblue2')
                    elif group == hops:
                        dot.node(column_id, column_name, style='filled', fillcolor='coral')
                    else:
                        dot.node(column_id, column_name)


            counter += 1
            if counter % 3 == 0:
                counter = 0
                dataset_name = ''
                table_name = ''
                group += 1
                # establish link between column, tables, dataset
                if column_id not in nodes:
                    dot.edge(column_id, table_id, 'partOf')
                    dot.edge(table_id, dataset_id, 'PartOf')
                    nodes.append(column_id)
        # establish content simi
        for j in range(len(similar) - 1):
            dot.edge(similar[j], similar[j + 1], 'similar', dir='none')
    dot.attr(label='Paths between starting nodes in blue and target nodes in orange')
    return dot


def generate_graphviz(df: pd.DataFrame, predicate: str):
    def parse_starting_or_target_nodes(dot, row, column_ids: list, table_ids: list, dataset_ids: list, start: bool) -> str :
        relation_name = 'partOf'
        if start:
            dataset_name = row[0]
            table_name = row[1]
            column_name = row[3]
            color = 'lightblue2'
        else:
            dataset_name = row[-4]
            table_name = row[-3]
            column_name = row[-1]
            color = 'darkorange3'
        dataset_id = str(generate_component_id(dataset_name))
        table_id = str(generate_component_id(dataset_name, table_name))
        column_id = str(generate_component_id(dataset_name, table_name, column_name))
        if column_id in column_ids:
            return column_id
        dot.node(column_id, column_name, style='filled', fillcolor=color)
        column_ids.append(column_id)
        if table_id in table_ids:
            dot.edge(column_id, table_id, relation_name)
            return column_id
        dot.node(table_id, table_name, style='filled', fillcolor=color)
        table_ids.append(table_id)
        if dataset_id in dataset_ids:
            dot.edge(column_id, table_id, relation_name)
            dot.edge(table_id, dataset_id, relation_name)
            return column_id
        dot.node(dataset_id, dataset_name, style='filled', fillcolor=color)
        dataset_ids.append(dataset_id)
        dot.edge(column_id, table_id, relation_name)
        dot.edge(table_id, dataset_id, relation_name)
        return column_id

    def parse_intermediate_nodes(dot, row, column_ids: list, table_ids: list, dataset_ids: list) -> list:
        ids = []
        relation_name = 'partOf'
        for i in range(4, len(row) - 4, 5):
            dataset_name = row[i]
            table_name = row[i + 1]
            land_in_column_name = row[i + 2]
            take_off_column_name = row[i + 4]
            dataset_id = str(generate_component_id(dataset_name))
            table_id = str(generate_component_id(dataset_name, table_name))
            land_in_column_id = str(generate_component_id(dataset_name, table_name, land_in_column_name))
            take_off_column_id = str(generate_component_id(dataset_name, table_name, take_off_column_name))
            ids.extend([land_in_column_id, take_off_column_id])

            land_in_column_exist = False
            take_off_column_exist = False
            if land_in_column_id in column_ids:
                land_in_column_exist = True
            dot.node(land_in_column_id, land_in_column_name)
            column_ids.append(land_in_column_id)

            if take_off_column_id in column_ids:
                take_off_column_exist = True

            if land_in_column_exist and take_off_column_exist:
                continue
            dot.node(take_off_column_id, take_off_column_name)
            column_ids.append(take_off_column_id)

            if table_id in table_ids:
                if land_in_column_id == take_off_column_id:
                    dot.edge(land_in_column_id, table_id, relation_name)
                else:
                    dot.edge(land_in_column_id, table_id, relation_name)
                    dot.edge(take_off_column_id, table_id, relation_name)
                continue

            dot.node(table_id, table_name)
            table_ids.append(table_id)
            if dataset_id in dataset_ids:
                if land_in_column_id == take_off_column_id:
                    dot.edge(land_in_column_id, table_id, relation_name)
                else:
                    dot.edge(land_in_column_id, table_id, relation_name)
                    dot.edge(take_off_column_id, table_id, relation_name)
                dot.edge(table_id, dataset_id, relation_name)
                continue
            dot.node(dataset_id, dataset_name)
            dataset_ids.append(dataset_id)
            if land_in_column_id == take_off_column_id:
                dot.edge(land_in_column_id, table_id, relation_name)
            else:
                dot.edge(land_in_column_id, table_id, relation_name)
                dot.edge(take_off_column_id, table_id, relation_name)
            dot.edge(table_id, dataset_id, relation_name)
        return ids

    def establish_relationships(dot, row_ids: list, relationships: list):

        for j in range(0, len(row_ids) - 1, 2):
            pair = (row_ids[j], row_ids[j + 1])
            if pair[0] == pair[1]:
                continue
            if not pair in relationships:
                relationships.append(pair)
                dot.edge(pair[0], pair[1], 'similar', dir='none')

    col_ids = []
    tab_ids = []
    data_ids = []
    relations = []
    dot_graph = Digraph(strict=True)
    for i in range(len(df)):
        r = df.iloc[i]
        row_col_ids = []
        starting_column_id = parse_starting_or_target_nodes(dot_graph, r, col_ids, tab_ids, data_ids, True)
        intermediate_col_ids = parse_intermediate_nodes(dot_graph, r, col_ids, tab_ids, data_ids)
        target_col_id = parse_starting_or_target_nodes(dot_graph, r, col_ids, tab_ids, data_ids, False)

        row_col_ids.append(starting_column_id)
        row_col_ids.extend(intermediate_col_ids)
        row_col_ids.append(target_col_id)

        establish_relationships(dot_graph, row_col_ids, relations)
    dot_graph.attr(label='Paths between starting nodes in blue and target nodes in orange', size='8,75,10')

    return dot_graph
