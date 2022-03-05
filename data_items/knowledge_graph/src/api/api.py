import itertools

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from enums.relation import Relation
from storage.kwtype import KWType
from utils import generate_label, generate_component_id, generate_graphviz
from word_embedding.embeddings_client import n_similarity


def _generate_path_of_df_columns(relation):
    return ['pre_nid', 'pre_db_name', 'pre_file_name', 'pre_column_name', 'pre_data_type',
            'target_nid', 'target_db_name', 'target_file_name', 'target_column_name',
            'target_data_type', relation.name + '_score']


class API:

    def __init__(self, elasticClient, rdfClient):
        self.elasticClient = elasticClient
        self.rdfClient = rdfClient
        self.df_columns = ['column_name', 'table_name', 'dataset_name', 'number_of_distinct_values', 'number_of_values',
                           'number_of_missing_values', 'origin', 'cardinality', 'minimum_value', 'maximum_value',
                           'median', 'column_data_type']

    def get_number_of_datasets(self, show_query=False):
        data = self.rdfClient.get_number_of_datasets(show_query=show_query)
        df = pd.DataFrame(list(data), columns=['number_of_datasets'])
        return df

    def get_tables_in(self, dataset_name, max_results=15, show_query=False):
        data = self.rdfClient.get_tables_in(dataset_name, max_results=max_results, show_query=show_query)
        df = pd.DataFrame(list(data), columns=['db_name', 'column_name'])
        return df

    def get_all_tables(self, show_query=False):
        data = self.rdfClient.get_all_tables(show_query=show_query)
        df = pd.DataFrame(list(data), columns=['db_name', 'column_name'])
        return df

    # search columns having labels by a regex
    def search_columns(self, keyword: str, show_query: bool = False) -> pd.DataFrame:
        data = self.rdfClient.search_columns(keyword, show_query)
        df = pd.DataFrame(list(data), columns=self.df_columns)
        return df

    # search columns having name by a regex
    def search_table_by_name(self, table_name: str, show_query: bool = False) -> pd.DataFrame:
        data = self.rdfClient.search_table_by_name(table_name, show_query)
        df = pd.DataFrame(list(data), columns=['table_name', 'dataset_name', 'origin', 'number_of_columns',
                                               'number_of_rows', 'path'])
        return df

    # search columns having labels by a regex
    def search_tables(self, keyword: str, show_query: bool = False) -> pd.DataFrame:
        data = self.rdfClient.search_tables(keyword, show_query)
        df = pd.DataFrame(list(data), columns=['table_name', 'dataset_name', 'origin', 'number_of_columns',
                                               'number_of_rows', 'path'])
        return df

    def search_tables_on(self, conditions: list, show_query: bool = False) -> pd.DataFrame:
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
                sts = '?column' + str(i) + ' rdf:type lac:column.' \
                                           '\n?column' + str(i) + ' dct:isPartOf ?table.' \
                                                                  '\n?column' + str(i) + ' lac:origin ?origin.' \
                                                                                         '\n?column' + str(
                    i) + ' rdfs:label ?label' + str(i) + '.'
                statements.append(sts)
                or_conditions = '|'.join(t)
                regex = 'regex(?label' + str(i) + ', "' + or_conditions + '", "i")'
                filters.append(regex)
                i += 1
            return '\n'.join(statements), ' && '.join(filters)

        data = self.rdfClient.search_tables_on(parsed_conditions(conditions), show_query)
        df = pd.DataFrame(list(data), columns=['table_name', 'dataset_name', 'origin', 'number_of_columns',
                                               'number_of_rows', 'path'])
        return df

    # given a list of values, find similar columns
    # TO-DO also support Numerical
    def get_joinable_columns(self, data, show_query=False):
        def _create_minhashLSH():
            result = self.elasticClient.get_profiles_minhash()
            content_index = MinHashLSH(threshold=0.6, num_perm=512, weights=(0.2, 0.8))
            for id, mh_sig in result:
                mh_obj = MinHash(num_perm=512)
                mh_array = np.asarray(mh_sig, dtype=int)
                mh_obj.hashvalues = mh_array
                content_index.insert(id, mh_obj)
            return content_index

        def _get_similar_cols(user_values: list, indexer) -> list:
            values_MH = MinHash(num_perm=512)
            for v in user_values:
                if isinstance(v, str):
                    values_MH.update(v.lower().encode('utf8'))

            res = indexer.query(values_MH)
            return res
        if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.core.series.Series)):
            raise TypeError('data should be either a pandas DataFrame or pandas Series')
        df_list = []
        if isinstance(data, pd.core.series.Series):
            data = pd.DataFrame(data)
        if data.empty:
            return pd.DataFrame([], columns=self.df_columns)
        content_indexer = _create_minhashLSH()
        for col_name in data.select_dtypes(include=['object']).columns:
            ids = _get_similar_cols(data[col_name], content_indexer)
            data = list(self.rdfClient.get_joinable_columns(ids, show_query))
            original_column_data = [col_name for _ in range(len(data))]
            df = pd.DataFrame(list(data), columns=self.df_columns)
            df.insert(0, 'original_column', original_column_data, True)
            df_list.append(df)
        if not df_list:
            return pd.DataFrame(df_list, self.df_columns)
        return pd.concat(df_list, ignore_index=True)

    def get_shortest_path_between_columns(self, col1_info: pd.core.series.Series, col2_info: pd.core.series.Series,
                                          via: str = 'pkfk', max_hops: int = 5, show_query: bool = False):
        # check if they are pandas series
        if not (isinstance(col1_info, pd.core.series.Series) and isinstance(col2_info, pd.core.series.Series)):
            raise TypeError('col1_info and col2_info should be pandas series')
        col1_info_index = col1_info.index
        col2_info_index = col2_info.index
        if not (
                'column_name' in col1_info_index and 'table_name' in col1_info_index and 'dataset_name' in col1_info_index):
            raise ValueError('col1_info index has to contain ["column_name", "table_name", "dataset_name"]')
        if not (
                'column_name' in col2_info_index and 'table_name' in col2_info_index and 'dataset_name' in col2_info_index):
            raise ValueError('col2_info index has to contain ["column_name", "table_name", "dataset_name"]')

        if not (via == 'pkfk' or via == 'semanticSimilarity' or via == 'contentSimilarity'):
            raise ValueError('via only takes [pkfk, semanticSimilarity, contentSimilarity]')

        in_id = generate_component_id(col1_info['dataset_name'], col1_info['table_name'], col1_info['column_name'])

        target_id = generate_component_id(col2_info['dataset_name'], col2_info['table_name'], col2_info['column_name'])

        data = self.rdfClient.get_shortest_path_between_columns(in_id, target_id, via, max_hops, show_query)
        df = pd.DataFrame(list(data), columns=self.df_columns)
        return df

    # based on string values
    # to do support numerical values
    def get_joinable_columns_between(self, table1: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
        def _create_minhash(values: pd.Series):
            values_mh = MinHash(num_perm=512)
            for v in values:
                if isinstance(v, str):
                    values_mh.update(v.lower().encode('utf8'))
            return values_mh

        if not (isinstance(table1, pd.DataFrame) and isinstance(table2, pd.DataFrame)):
            raise TypeError('The passed arguments should be of type pandas dataframes')
        joinable_columns = []

        # index first table:
        indexer = MinHashLSH(threshold=0.6, num_perm=512, weights=(0.2, 0.8))
        for colname in table1.select_dtypes(include=['object']).columns:
            mh = _create_minhash(table1[colname])
            indexer.insert(colname, mh)

        # query the second table
        for colname in table2.select_dtypes(include=['object']).columns:
            mh = _create_minhash(table2[colname])
            cols = indexer.query(mh)
            joinable_columns.extend([(col, colname) for col in cols])
        df = pd.DataFrame(joinable_columns, columns=['First Table Columns', 'Second Table Columns'])
        return df

    def get_unionable_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, sim_threshold: float = 0.5) -> pd.DataFrame:
        def _drop_duplicates(cn1: list, cn2: list):
            duplicates = set(cn1).intersection(set(cn2))
            for d in duplicates:
                cn1.remove(d)
                cn2.remove(d)
            return cn1, cn2

        def _create_combinations(colname1: str, colname2: str) -> pd.DataFrame:
            colname1_tokens = colname1.split(' ')
            colname2_tokens = colname2.split(' ')
            if len(colname1_tokens) > 1 and len(colname2_tokens) > 1:
                colname1_tokens, colname2_tokens = _drop_duplicates(colname1_tokens, colname2_tokens)
            combs = itertools.product(colname1_tokens, colname2_tokens)
            return list(combs)

        def _calculate_similarity(comb: list):
            if not comb:
                return 1.0
            similarity_sum = 0
            for t1, t2 in combinations:
                similarity_sum += n_similarity([str(t1)], [str(t2)])
            return similarity_sum / len(comb)

        if not (isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)):
            raise TypeError('The inputs have to be of type pandas datarames')
        matched = []
        for c1, c2 in itertools.product(df1.columns, df2.columns):
            c1_label = generate_label(c1, 'en').get_text()
            c2_label = generate_label(c2, 'en').get_text()
            combinations = _create_combinations(c1_label, c2_label)
            similarity = _calculate_similarity(combinations)
            if similarity >= sim_threshold:
                matched.append((c1, c2))
        unionable_df = pd.DataFrame(matched, columns=['First dataframe columns', 'Second dataframe columns'])
        return unionable_df

    def _add_annotation(self, annotation: str, dataset_name: str, table_name: str, predicate: str,
                        show_query: bool = False):
        if table_name is None:
            iri = '<' + self.rdfClient.get_iri_of_dataset(dataset_name) + '>'
        else:
            iri = '<' + self.rdfClient.get_iri_of_table(dataset_name, table_name) + '>'
        num_annotations = self.rdfClient.get_num_of_annotations_in(iri, predicate=predicate)
        if num_annotations == 0:
            self.rdfClient.add_first_element(iri, annotation, predicate, show_query=show_query)
        else:
            self.rdfClient.add_element(iri, annotation, predicate, num_annotations, show_query=show_query)

    def add_dataset_usage(self, dataset_name: str, used_in: str, show_query: bool = False):
        self._add_annotation('<' + used_in + '>', dataset_name, None, 'lac:used_in', show_query)

    def add_table_usage(self, dataset_name: str, table_name: str, used_in: str, show_query: bool = False):
        self._add_annotation('<' + used_in + '>', dataset_name, table_name, 'lac:used_in', show_query)

    def add_dataset_insight(self, dataset_name: str, insight: str, show_query: bool = False):
        self._add_annotation('"' + insight + '"', dataset_name, None, 'lac:insights', show_query)

    def add_table_insight(self, dataset_name: str, table_name: str, insight: str, show_query: bool = False):
        self._add_annotation('"' + insight + '"', dataset_name, table_name, 'lac:insights', show_query)

    def _get_annotatios(self, dataset_name: str, table_name: str, predicate: str, show_query: bool = False):
        if table_name is None:
            iri = self.rdfClient.get_iri_of_dataset(dataset_name)
            if not iri:
                raise ValueError(dataset_name + ' is not found')
        else:
            iri = self.rdfClient.get_iri_of_table(dataset_name, table_name)
            if not iri:
                raise ValueError(table_name + ' is not found')
        data = self.rdfClient.get_usages('<' + iri + '>', predicate, show_query)
        df = pd.DataFrame(data, columns=['annotation'])
        return df

    def get_dataset_usages(self, dataset_name: str, show_query: bool = False):
        return self._get_annotatios(dataset_name, None, 'lac:used_in', show_query)

    def get_table_usages(self, dataset_name: str, table_name: str, show_query: bool = False):
        return self._get_annotatios(dataset_name, table_name, 'lac:used_in', show_query)

    def get_dataset_insights(self, dataset_name: str, show_query: bool = False):
        return self._get_annotatios(dataset_name, None, 'lac:insights', show_query)

    def get_table_insights(self, dataset_name: str, table_name: str, show_query: bool = False):
        return self._get_annotatios(dataset_name, table_name, 'lac:insights', show_query)

    def get_path_between_tables(self, starting_table_info: pd.core.series.Series,
                                target_table_info: pd.core.series.Series , hops: int,
                                predicate: str = 'lac:contentSimilarity', show_query: bool = False):
        if not isinstance(starting_table_info, pd.core.series.Series ):
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

        starting_table_iri = self.rdfClient.get_iri_of_table(dataset_name=generate_label(starting_dataset_name, 'en').get_text(),
                                                             table_name=generate_label(starting_table_name, 'en').get_text())
        target_table_iri = self.rdfClient.get_iri_of_table(dataset_name=generate_label(target_dataset_name, 'en').get_text(),
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

    def execute_sparql_query(self, query: str, method: str = 'get'):
        pass

    ##################
    def search(self, keyword: str, kwType: KWType, max_results: int, approximate=False, show_query=False):
        if approximate:
            if kwType == KWType.KW_CONTENT:
                data = self.elasticClient.search_keywords(keyword, kwType, max_results, show_query)
            elif kwType == KWType.KW_TABLE or kwType == KWType.KW_SEMANTIC:
                data = self.rdfClient.approximate_search(keyword, kwType, max_results, show_query)
        else:
            if kwType == KWType.KW_CONTENT:
                data = self.elasticClient.exact_search_keywords(keyword, kwType, max_results, show_query)
            elif kwType == KWType.KW_TABLE or kwType == KWType.KW_SEMANTIC:
                data = self.rdfClient.search(keyword, kwType, max_results, show_query)
        df = pd.DataFrame(list(data), columns=['nid', 'db_name', 'file_name', 'column_name', 'data_type']) \
            .drop_duplicates()
        return df

    def search_content(self, kw: str, approximate=False, max_results=15, show_query=False):
        if approximate:
            return self.search(kw, kwType=KWType.KW_CONTENT, max_results=max_results, approximate=True,
                               show_query=show_query)
        return self.search(kw, kwType=KWType.KW_CONTENT, max_results=max_results, show_query=show_query)

    def search_attribute(self, kw: str, approximate=False, max_results=15, show_query=False):
        if approximate:
            return self.search(kw, kwType=KWType.KW_SEMANTIC, max_results=max_results, approximate=True,
                               show_query=show_query)
        return self.search(kw, kwType=KWType.KW_SEMANTIC, max_results=max_results, show_query=show_query)

    def search_table(self, kw: str, approximate=False, max_results=15, show_query=False):
        if approximate:
            return self.search(kw, kwType=KWType.KW_TABLE, max_results=max_results, approximate=True,
                               show_query=show_query)
        return self.search(kw, kwType=KWType.KW_TABLE, max_results=max_results, show_query=show_query)

    def get_related_to(self, keyword, relation: Relation, max_results: int, show_query=False):
        data = self.rdfClient.get_neighbors(keyword, relation, max_results, show_query=show_query)
        result = pd.DataFrame(list(data),
                              columns=['nid', 'db_name', 'file_name', 'column_name', 'data_type', 'score']) \
            .drop_duplicates()
        return result

    def get_content_similar_to(self, general_input, max_results=15, show_query=False):
        return self.get_related_to(general_input, Relation.contentSimilarity, max_results, show_query=show_query)

    def get_semantically_similar_to(self, general_input, max_results=15, show_query=False):
        return self.get_related_to(general_input, Relation.semanticSimilarity, max_results, show_query=show_query)

    def get_pkfk_of(self, general_input, max_results=15, show_query=False):
        return self.get_related_to(general_input, Relation.pkfk, max_results=max_results, show_query=show_query)

    """
    Combiner API
    """

    def get_intersection_between(self, a, b, table_mode=False, a_suffix='_1', b_suffix='_2'):
        """
        Returns elements that are both in a and b
        :param a: dataframe
        :param b: dataframe
        :param table_mode: get the intersection in Table mode
        :param b_suffix: in case of table mode, suffix of first df, default _1
        :param a_suffix: in case of table mode, suffix of first col, default _2
        :return: the intersection of the two provided iterable objects
        """
        if table_mode:
            result = pd.merge(a, b, how='inner', on=['file_name'], suffixes=(a_suffix, b_suffix))
        else:
            if 'score' in a.columns:
                a = a.drop(['score'], axis=1)
            if 'score' in b.columns:
                b = b.drop(['score'], axis=1)
            result = pd.merge(a, b, how='inner')
        return result

    def get_union_between(self, a, b):
        """
        Returns elements that are in either a or b
        :param a: dataframe
        :param b: dataframe
        :return: the union of the two provided iterable objects
        """

        return pd.concat([a, b]).drop_duplicates().reset_index(drop=True)

    def get_difference_between(self, a, b):
        """
        Returns elements that are in either a or b
        :param a: an iterable object
        :param b: another iterable object
        :return: the union of the two provided iterable objects
        """
        return pd.concat([a, b, b]).drop_duplicates(keep=False)

    def get_path_between_nodes(self, a, b, relation=Relation.pkfk, maxHops=5, show_query=False):
        def validate_input(param):
            if isinstance(param, pd.core.series.Series):
                return dict(param)
            elif isinstance(param, dict):
                return param
            elif isinstance(param, pd.core.frame.DataFrame) and len(param) == 1:
                return dict(param.iloc[0])
            else:
                raise TypeError('input has to be either dict or series')

        a = validate_input(a)
        b = validate_input(b)
        data = self.rdfClient.get_path_between_nodes(a, b, relation, maxHops, show_query=show_query)
        return pd.DataFrame(list(data), columns=_generate_path_of_df_columns(relation))

    def get_path_between_dfs(self, df1, df2=None, relation=Relation.pkfk, maxHops=5):
        if not isinstance(df1, pd.core.frame.DataFrame):
            raise TypeError('df1 has to be of type Dataframe')
        if not (isinstance(df2, pd.core.frame.DataFrame) or df2 is None):
            raise TypeError('df2 has to be either none or of type Dataframe')

        resultList = []
        if df2 is None:
            items = [(df1.iloc[i - 1], df1.iloc[j]) for i in range(1, len(df1)) for j in
                     range(i, len(df1))]
        else:
            l1 = [df1.iloc[i] for i in range(len(df1))]
            l2 = [df2.iloc[i] for i in range(len(df2))]
            items = itertools.product(l1, l2)
        for h1, h2 in items:
            i_df = self.get_path_between_nodes(h1, h2, relation, maxHops)
            if not i_df.empty:
                resultList.append(i_df)
        if not resultList:
            return pd.DataFrame(columns=_generate_path_of_df_columns(relation))
        return pd.concat(resultList, keys=[i for i in range(len(resultList))])
