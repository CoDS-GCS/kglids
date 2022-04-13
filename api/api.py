import stardog
from api.template import *
from api.helpers.helper import *
# from data_items.knowledge_graph.src.utils import generate_label, generate_component_id, generate_graphviz
from data_items.kglids_evaluations.src.helper.config import connect_to_blazegraph


class KGLiDS:
    def __init__(self, blazegraph_port=9999, blazegraph_namespace: str = 'soen6111'):
        self.config = connect_to_blazegraph(blazegraph_port, blazegraph_namespace)
        self.conn = connect_to_stardog()
        self.conn.begin()

    def get_datasets(self, show_query: bool = False):
        return get_datasets(self.config, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

    def get_tables(self, dataset: str = '', show_query: bool = False):
        if not dataset:
            print('Showing all available table(s): ')
        else:
            print("Showing table(s) for '{}' dataset: ".format(dataset))
        return get_tables(self.config, dataset, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

    def recommend_k_joinable_tables(self, table: pd.Series, k: int = 5, show_query: bool = False):
        if not isinstance(table, pd.Series):
            raise ValueError("table needs to be a type 'pd.Series'")
        # if not isinstance(dataset, str):
        #     raise ValueError("dataset needs to be a type 'str'")
        if not isinstance(k, int):
            raise ValueError("k needs to be a type 'int'")
        elif isinstance(table, pd.Series) and isinstance(k, int):
            dataset = table["Dataset"]
            table = table["Table"]
            recommendations = recommend_tables(self.config, dataset, table, k, 'data:hasPrimaryKeyForeignKeySimilarity',
                                               show_query)
            print('Showing the top-{} joinable table recommendations:'.format(len(recommendations)))
            return recommendations

    def recommend_k_unionable_tables(self, table: pd.Series, k: int = 5, show_query: bool = False):
        if not isinstance(table, pd.Series):
            raise ValueError("table needs to be a type 'pd.Series'")
        # if not isinstance(dataset, str):
        #     raise ValueError("dataset needs to be a type 'str'")
        if not isinstance(k, int):
            raise ValueError("k needs to be a type 'int'")
        elif isinstance(table, pd.Series) and isinstance(k, int):
            dataset = table["Dataset"]
            table = table["Table"]
            recommendations = recommend_tables(self.config, dataset, table, k, 'data:hasSemanticSimilarity',
                                               show_query)
            print('Showing the top-{} unionable table recommendations:'.format(len(recommendations)))
            return recommendations

    def get_table_info(self, table, show_query: bool = False):
        dataset = table["Dataset"]
        if 'Recommended_table' in table.keys():
            table = table["Recommended_table"]
        else:
            table = table["Table"]
        return get_table_info(self.config, dataset, table, show_query)

    def show_graph_info(self, show_query: bool = False):
        print('Information captured: ')
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
        print('Showing recommendations as per the following conditions:\nCondition = ', conditions)
        return pd.DataFrame(list(data), columns=['Dataset', 'Table', 'Number_of_columns',
                                                 'Number_of_rows', 'Path_to_table']).sort_values('Table',
                                                                                                 ignore_index=True,
                                                                                                 ascending=False)

    def get_path_between_tables(self, source_table: pd.Series, target_table: pd.Series, hops: int,
                                relation: str = 'data:hasPrimaryKeyForeignKeySimilarity', show_query: bool = False):
        return get_path_between_tables(self.config, source_table, target_table, hops, relation, show_query)

    def get_unionable_columns(self, df1, df2, thresh: float = 0.50):
        return get_unionable_columns(self.config, df1, df2, thresh)

    def query(self, rdf_query: str):
        return query_kglids(self.config, rdf_query)

    def get_top_scoring_ml_model(self, dataset: str = '', show_query=False):
        return get_top_scoring_ml_model(self.conn, dataset, show_query)

    def get_pipelines(self, author: str = '', show_query=False):
        return get_pipelines(self.conn, author, show_query).sort_values('Number_of_votes', ignore_index=True,
                                                                        ascending=False)

    def get_most_recent_pipeline(self, dataset: str = '', show_query=False):
        return get_most_recent_pipeline(self.conn, dataset, show_query)

    def get_top_k_scoring_pipelines_for_dataset(self, dataset: str = '', k: int = None, show_query=False):
        return get_top_k_scoring_pipelines_for_dataset(self.conn, dataset, k, show_query)

    def get_most_popular_parameters(self, library: str, parameters='all'):
        pass

    def search_classifier(self, dataset: str, show_query=False):
        return search_classifier(self.conn, dataset, show_query)

    def get_classifier(self, classifier: pd.Series, show_query=False):
        pipeline_name = classifier['Pipeline']
        classifier = classifier['Classifier']
        return get_classifier(self.conn, pipeline_name, classifier, show_query)

    def get_library_usage(self, dataset: str = '', show_query=False):
        return get_library_usage(self.conn, dataset, show_query)
