import pandas as pd
from api.template import *
# from data_items.knowledge_graph.src.utils import generate_label, generate_component_id, generate_graphviz
from data_items.kglids_evaluations.src.helper.config import connect_to_blazegraph


class KGLiDS:
    def __init__(self, port=7777, namespace: str = 'kglids_smallerReal'):
        self.config = connect_to_blazegraph(port, namespace)

    def get_datasets(self, show_query: bool = False):
        return get_datasets(self.config, show_query)

    def get_tables(self, dataset: str = '', show_query: bool = False):
        return get_tables(self.config, dataset, show_query)

    def recommend_k_tables(self, table: str, k: int = 5, show_query: bool = False):
        if not isinstance(table, str):
            raise ValueError("table needs to be a type 'str'")
        if not isinstance(k, int):
            raise ValueError("k needs to be a type 'int'")
        elif isinstance(table, str) and isinstance(k, int):
            recommendations = recommend_tables(self.config, table, k, show_query)
            print('Showing the top {} table recommendations!'.format(len(recommendations)))
            return recommendations

    def show_graph_info(self, show_query: bool = False):
        return show_graph_info(self.config, show_query)

    def search_tables_on(self, conditions: list, show_query: bool = False):

        def parsed_conditions(user_conditions):
            error_message = 'conditions need to be in encapsulated in list.\n' \
                            'lists in the list are associated by an \"and\" condition.\n' \
                            'String in each tuple will be joined by an \"or\" condition.\n' \
                            ' For instance [[a,b],[c]]'
            if not isinstance(user_conditions, list):
                raise TypeError(error_message)
            else:
                for l in user_conditions:
                    if not isinstance(l, list):
                        raise TypeError(error_message)
                    else:
                        for s in l:
                            if not isinstance(s, str):
                                raise TypeError(error_message)

            i = 1
            filters = []
            statements = []
            for t in user_conditions:
                sts = '?column' + str(i) + ' rdf:type kglids:Column.' \
                                           '\n?column' + str(i) + ' kglids:isPartOf ?table.' \
                                                                  '\n?column' + str(
                    i) + ' rdfs:label ?label' + str(i) + '.'

                statements.append(sts)
                or_conditions = '|'.join(t)
                regex = 'regex(?label' + str(i) + ', "' + or_conditions + '", "i")'
                filters.append(regex)
                i += 1
            return '\n'.join(statements), ' && '.join(filters)

        data = search_tables_on(self.config, parsed_conditions(conditions), show_query)
        return pd.DataFrame(list(data), columns=['table_name', 'dataset_name', 'number_of_columns',
                                                 'number_of_rows', 'path']).sort_values('number_of_rows',
                                                                                        ignore_index=True,
                                                                                        ascending=False)
"""
    def get_path_between_tables(self, starting_table_info: pd.core.series.Series,
                                target_table_info: pd.core.series.Series, hops: int,
                                predicate: str = 'lac:pkfk', show_query: bool = False):

        def get_iri_of_table(dataset_name: str, table_name: str, show_query: bool = False):
            dataset_label = generate_label(dataset_name, 'en')
            table_label = generate_label(table_name, 'en')
            return _get_iri(dataset_label, table_label, show_query)
        
        
        if not isinstance(starting_table_info, pd.core.series.Series):
            raise TypeError('starting_table_info should be a series')
        if not isinstance(target_table_info, pd.core.series.Series):
            raise TypeError('target_table_info should be a series')
        if not ('dataset_name' in starting_table_info.index and 'table_name' in starting_table_info.index):
            raise ValueError('starting table info should contain dataset_name and table_name')
        if not ('dataset_name' in target_table_info.index and 'table_name' in target_table_info.index):
            raise ValueError('target table info should contain dataset_name and table_name')

        starting_dataset_name = starting_table_info['dataset_name']
        starting_table_name = starting_table_info['table_name']

        target_dataset_name = target_table_info['dataset_name']
        target_table_name = target_table_info['table_name']

        starting_table_iri = get_iri_of_table(
            dataset_name=generate_label(starting_dataset_name, 'en').get_text(),
            table_name=generate_label(starting_table_name, 'en').get_text())
        target_table_iri = get_iri_of_table(
            dataset_name=generate_label(target_dataset_name, 'en').get_text(),
            table_name=generate_label(target_table_name, 'en').get_text())
        if starting_table_iri is None:
            raise ValueError(str(starting_table_info) + ' does not exist')
        if target_table_iri is None:
            raise ValueError(str(target_table_info) + ' does not exist')

        data = self.rdfClient.get_paths_between_tables('<' + starting_table_iri + '>', '<' + target_table_iri + '>',
                                                       predicate, hops, show_query)
        path_row = ['starting_dataset', 'starting_table', 'starting_table_path', 'starting_column']
        for i in range(2, hops + 1):
            intermediate = ['intermediate_dataset' + str(i), 'intermediate_table' + str(i),
                            'intermediate_column_land_in' + str(i), 'intermediate_table_path' + str(i),
                            'intermediate_column_take_off' + str(i)]
            path_row.extend(intermediate)
        path_row.extend(['target_dataset', 'target_table', 'target_table_path', 'target_column'])
        df = pd.DataFrame(list(data), columns=path_row)
        dot = generate_graphviz(df, predicate)
        return dot
"""
