from api.template import *
from api.helpers.helper import *
from collections import Counter
from tqdm import tqdm


class KGLiDS:
    def __init__(self, endpoint: str = 'localhost', port=5820, db: str = 'kglids'):
        self.conn = connect_to_stardog(endpoint, port, db)
        self.conn.begin()

    def get_datasets_info(self, show_query: bool = False):
        return get_datasets_info(self.conn, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

    def get_tables_info(self, dataset: str = '', show_query: bool = False):
        if not dataset:
            print('Showing all available table(s): ')
        else:
            print("Showing table(s) for '{}' dataset: ".format(dataset))
        return get_tables_info(self.conn, dataset, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

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
            recommendations = recommend_tables(self.conn, dataset, table, k, 'data:hasPrimaryKeyForeignKeySimilarity',
                                               show_query)
            recommendations['Score'] = list(map(lambda x: round(x, 2), [float(i) / max(recommendations['Score'].
                                                       tolist()) for i in (recommendations['Score'].tolist())]))

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
            recommendations = recommend_tables(self.conn, dataset, table, k, 'data:hasSemanticSimilarity',
                                               show_query)
            recommendations['Score'] = list(map(lambda x: round(x, 2), [float(i) / max(recommendations['Score'].
                                                tolist()) for i in (recommendations['Score'].tolist())]))

            print('Showing the top-{} unionable table recommendations:'.format(len(recommendations)))
            return recommendations

    def get_table_info(self, table, show_query: bool = False):
        dataset = table["Dataset"]
        if 'Recommended_table' in table.keys():
            table = table["Recommended_table"]
        else:
            table = table["Table"]
        return get_table_info(self.conn, dataset, table, show_query)

    def show_graph_info(self, show_query: bool = False):
        print('Information captured: ')
        return show_graph_info(self.conn, show_query)

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

        data = search_tables_on(self.conn, parsed_conditions(conditions), show_query)
        print('Showing recommendations as per the following conditions:\nCondition = ', conditions)
        df = pd.DataFrame(list(data), columns=['Dataset', 'Table', 'Number_of_columns',
                                                 'Number_of_rows', 'Path_to_table']).sort_values('Number_of_rows',
                                                                                                 ignore_index=True,
                                                                                               ascending=False)
        df['Number_of_rows'] = df['Number_of_rows'].apply(lambda x: int(x))
        df['Number_of_columns'] = df['Number_of_columns'].apply(lambda x: int(x))
        return df

    def get_path_between_tables(self, source_table: pd.Series, target_table: pd.Series, hops: int,
                                relation: str = 'data:hasPrimaryKeyForeignKeySimilarity', show_query: bool = False):
        return get_path_between_tables(self.conn, source_table, target_table, hops, relation, show_query)

    def query(self, rdf_query: str):
        return query_kglids(self.conn, rdf_query)

    def get_top_scoring_ml_model(self, dataset: str = '', show_query=False):
        return get_top_scoring_ml_model(self.conn, dataset, show_query)

    def get_pipelines_info(self, author: str = '', show_query=False):
        return get_pipelines_info(self.conn, author, show_query).sort_values('Number_of_votes', ignore_index=True,
                                                                             ascending=False)

    def get_most_recent_pipeline(self, dataset: str = '', show_query=False):
        return get_most_recent_pipeline(self.conn, dataset, show_query)

    def get_top_k_scoring_pipelines_for_dataset(self, dataset: str = '', k: int = None, show_query=False):
        return get_top_k_scoring_pipelines_for_dataset(self.conn, dataset, k, show_query)

    def get_most_popular_parameters(self, library: str, parameters='all'):
        pass

    def search_classifier(self, dataset: str = '', show_query=False):
        return search_classifier(self.conn, dataset, show_query)

    def get_hyperparameters(self, classifier: pd.Series, show_query=False):
        pipeline_name = classifier['Pipeline']
        classifier = classifier['Classifier']
        return get_hyperparameters(self.conn, pipeline_name, classifier, show_query)

    def get_top_k_library_used(self, dataset: str = '', k: int = 5, show_query=False):
        return get_library_usage(self.conn, dataset, k, show_query)

    def get_top_used_libraries(self, k: int = 5, task: str = 'classification', show_query: bool = False):
        supported_tasks = ['classification', 'regression', 'visualization', 'clustering']
        if task not in supported_tasks:
            raise ValueError(' invalid task, try using one of the following tasks: \n'
                             'classification, regression, visualization or clustering!')
        else:
            library_info = get_top_used_libraries(self.conn, task, show_query)
            if len(library_info) == 0:
                print('No library found for {}'.format(task))
                return
            library_info['Module'] = library_info['Module'].apply(lambda module: module.replace('/', '.'))
            # fetch top k libraries by maximum occurrence
            libraries = library_info['Library']
            library_count = Counter(libraries)
            if k > len(library_count):
                k = len(library_count)
                if k == 1:
                    print('Single library was found for {}: '.format(k, task))
                else:
                    print('Maximum {} libraries were found for {}: '.format(k, task))
            else:
                if k == 1:
                    print('Showing the top used library for {}: '.format(task))
                else:
                    print('Showing the top {} libraries for {}: '.format(k, task))
            libraries = sorted(library_count, key=library_count.get, reverse=True)[:k]

            for i in libraries[:len(libraries)-1]:
                print(i + ',', end=' ')
            print(libraries[-1])
            for k, v in tqdm(library_info.to_dict('index').items()):
                if v['Library'] not in libraries:
                    library_info = library_info.drop(k)
            return library_info.sort_values(by=['Library']).reset_index(drop=True)

    def get_pipelines_calling_libraries(self, components: list, show_query: bool = False):
        return get_pipelines_calling_libraries(self.conn, components, show_query)

    def get_pipelines_for_deep_learning(self, show_query: bool = False):
        return get_pipelines_for_deep_learning(self.conn, show_query)

    def recommend_transformations(self, show_query: bool = False):
        return recommend_transformations(self.conn, show_query)
