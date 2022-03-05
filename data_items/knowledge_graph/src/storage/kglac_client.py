import config as c
import pandas as pd
import storage.query_templates as queryTemplates
from SPARQLWrapper import SPARQLWrapper, JSON
from storage.kwtype import KWType
from utils import generate_label


def get_id_from_uri(uri):
    return str(uri).split('lac#')[1]


def _make_values_of_ids(h):
    if isinstance(h, dict):
        return 'lac:' + str(h['nid'])
    elif isinstance(h, str):
        if not h.isdigit():
            raise ValueError('keyword must be an id, hit, or a DRS ')
        return 'lac:' + h
    elif isinstance(h, pd.core.frame.DataFrame):
        return ' '.join(['lac:' + str(i) for i in h['nid']])
    elif isinstance(h, list):
        return ' '.join(['lac:' + str(id) for id in h])
    else:
        raise TypeError


def _print_query(query):
    print('*****************************QUERY*****************************')
    print('-----------------------------START-----------------------------')
    print(query)
    print('------------------------------END------------------------------')


def _create_column_df_row(result):
    minimum_value = ''
    maximum_value = ''
    median_value = ''
    if result['type']['value'] == 'N':
        minimum_value = float(result['min']['value'])
        maximum_value = float(result['max']['value'])
        median_value = float(result['median']['value'])

    return {'column_name': result['name']['value'],
            'table_name': result['table_name']['value'],
            'dataset_name': result['dataset_name']['value'],
            'number_of_distinct_values': float(result['distinct']['value']),
            'number_of_values': float(result['total']['value']),
            'number_of_missing_values': float(result['missing']['value']),
            'origin': result['origin']['value'],
            'cardinality': float(result['cardinality']['value']),
            'minimum_value': minimum_value,
            'maximum_value': maximum_value,
            'median': median_value,
            'column_data_type': result['type']['value']
            }


def _create_tables_df_row(results):
    return {
        'table_name': results['name']['value'],
        'dataset_name': results['dataset_name']['value'],
        'origin': results['origin']['value'],
        'number_of_columns': float(results['number_of_columns']['value']),
        'number_of_rows': float(results['number_of_rows']['value']),
        'path': results['path']['value']
    }


def _create_simple_df_row(result):
    return {'nid': get_id_from_uri(result['id']['value']),
            'db_name': result['dn']['value'],
            'file_name': result['fn']['value'],
            'column_name': result['cn']['value'],
            'data_type': result['dt']['value']}


def _create_path_row(result, hops):
    data = {'starting_column': result['c1name']['value'],
            'starting_table': result['t1name']['value'],
            'starting_table_path': result['t1path']['value'],
            'starting_dataset': result['d1name']['value']}

    intermediate = {}
    for i in range(2, hops + 1):
        intermediate.update({'intermediate_column_land_in' + str(i): result['c' + str(i) + 'name']['value'],
                             'intermediate_table' + str(i): result['t' + str(i) + 'name']['value'],
                             'intermediate_table_path' + str(i): result['t' + str(i) + 'path']['value'],
                             'intermediate_column_take_off' + str(i): result['cc' + str(i) + 'name']['value'],
                             'intermediate_dataset' + str(i): result['d' + str(i) + 'name']['value']})
    data.update(intermediate)

    data.update({'target_column': result['c' + str(hops + 1) + 'name']['value'],
                 'target_table': result['t' + str(hops + 1) + 'name']['value'],
                 'target_table_path': result['t' + str(hops + 1) + 'path']['value'],
                 'target_dataset': result['d' + str(hops + 1) + 'name']['value']})
    return data


def _create_get_neighbor_df_row(result):
    columns = _create_simple_df_row(result)
    columns.update({'score': float(result['score']['value'])})
    return columns


class KGLacClient:
    client = None

    def __init__(self, namespace):
        self.client = SPARQLWrapper(c.url + namespace + '/sparql')

    def process_query(self, query, queryReturnFormat, method: str = 'GET'):
        self.client.method = method
        self.client.setQuery(query)
        if method == 'GET':
            self.client.setReturnFormat(queryReturnFormat)
            return self.client.query().convert()
        self.client.query()

    def get_number_of_datasets(self, show_query=False):
        query = queryTemplates.get_number_of_datasets()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in bindings:
            yield {'number_of_datasets': int(result['count']['value'])}  # fix score later

    def get_tables_in(self, table_name, max_results=15, show_query=False):
        query = queryTemplates.get_tables_in_dataset() % (table_name, max_results)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in bindings:
            yield {'db_name': result['dn']['value'], 'column_name': result['cn']['value']}

    def get_all_tables(self, show_query=False):
        query = queryTemplates.get_all_tables()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in bindings:
            yield {'db_name': result['dn']['value'], 'column_name': result['cn']['value']}

    # search columns having labels by a regex
    def search_columns(self, keyword: str, show_query=False):
        query = queryTemplates.search_columns(keyword)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_column_df_row(result)

    # search tables having name a regex
    def search_table_by_name(self, keyword: str, show_query: bool = False):
        query = queryTemplates.search_table_by_name(keyword)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_tables_df_row(result)



    # search tables having labels a regex
    def search_tables(self, keyword: str, show_query: bool = False):
        query = queryTemplates.search_tables(keyword)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_tables_df_row(result)

    def search_tables_on(self, conditions: tuple, show_query: bool = False):
        query = queryTemplates.search_tables_on(conditions)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_tables_df_row(result)

    def get_joinable_columns(self, ids: list, show_query: bool = False):

        query = queryTemplates.get_joinable_columns(_make_values_of_ids(ids))
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_column_df_row(result)

    def get_shortest_path_between_columns(self, in_id: int, target_id: int, via: str = 'pkfk', max_hops: int = 5,
                                          show_query: bool = False):
        query = queryTemplates.get_shortest_path_between_columns(in_id, target_id, via, max_hops)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_column_df_row(result)

    def _get_iri(self, dataset_name: str, table_name: str = None, show_query: bool = False):
        if table_name is None:
            query = queryTemplates.get_iri_of_dataset(dataset_name)
        else:
            query = queryTemplates.get_iri_of_table(dataset_name, table_name)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return None
        return str(bindings[0]['id']['value'])

    def get_iri_of_dataset(self, dataset_name: str, show_query: bool = False):
        dataset_label = generate_label(dataset_name, 'en')
        return self._get_iri(dataset_label, None, show_query)

    def get_iri_of_table(self, dataset_name: str, table_name: str, show_query: bool = False):
        dataset_label = generate_label(dataset_name, 'en')
        table_label = generate_label(table_name, 'en')
        return self._get_iri(dataset_label, table_label, show_query)

    def get_num_of_annotations_in(self, iri: str, predicate: str, show_query: bool = False) -> int:
        query = queryTemplates.get_num_of_annotations_in(iri, predicate)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        num_annotations = int(results['results']['bindings'][0]['num_annotation']['value'])
        return num_annotations

    def add_first_element(self, iri: str, annotation: str, predicate: str, show_query: bool = False):
        query = queryTemplates.add_first_element(iri, annotation, predicate)
        if show_query:
            _print_query(query)
        self.process_query(query, JSON, method='POST')

    def add_element(self, iri: str, annotation: str, predicate: str, bag_length: int, show_query: bool = False):
        query = queryTemplates.add_element(iri, annotation, predicate, bag_length)
        if show_query:
            _print_query(query)
        self.process_query(query, JSON, method='POST')

    def get_usages(self, iri: str, predicate: str, show_query: bool = False):
        query = queryTemplates.get_usages(iri, predicate)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield {'annotation': result['annotation']['value']}

    def get_paths_between_tables(self, start_iri: str, target_iri: str, predicate: str, hops: int,
                                 show_query: bool = False):

        def _generate_starting_nodes() -> str:
            return '\n    ?c1 schema:name ?c1name.'  \
                '\n    ?c1 dct:isPartOf ?t1.' \
                '\n    ?t1 schema:name ?t1name.' \
                '\n    ?t1 lac:path ?t1path.' \
                '\n    ?t1 dct:isPartOf ?d1.' \
                '\n    ?d1 schema:name ?d1name.'

        def _generate_intermediate_nodes(h: int) -> str:
            inters = ''
            for i in range(2, h + 1):
                inter = '\n    ?c' + str(i) + ' a lac:column.' \
                    '\n    ?c' + str(i) + ' schema:name ?c' + str(i) + 'name.' \
                    '\n    ?c' + str(i) + ' dct:isPartOf ?t' + str(i) + '.' \
                    '\n    ?t' + str(i) + ' schema:name ?t' + str(i) + 'name.' \
                    '\n    ?t' + str(i) + ' lac:path ?t' + str(i) + 'path.' \
                    '\n    ?cc' + str(i) + ' dct:isPartOf ?t' + str(i)+ '.' \
                    '\n    ?cc' + str(i) + ' schema:name ?cc' + str(i) + 'name.' \
                    '\n    ?t' + str(i) + ' dct:isPartOf ?d' + str(i) + '.' \
                    '\n    ?d' + str(i) + ' schema:name ?d' + str(i) + 'name.'
                inters += inter
            return inters

        def _generate_target_nodes(h: int )-> str:
            return '\n    ?c' + str(h + 1) + ' schema:name ?c' + str(h + 1) + 'name.' \
                   '\n    ?c' + str(h + 1) + ' dct:isPartOf ?t' + str(h + 1) + '.' \
                   '\n    ?t' + str(h + 1) + ' schema:name ?t' + str(h + 1) + 'name.' \
                   '\n    ?t' + str(h + 1) + ' lac:path ?t' + str(h + 1) + 'path.' \
                   '\n    ?t' + str(h + 1) +  ' dct:isPartOf ?d' + str(h + 1) + '.' \
                   '\n    ?d' + str(h + 1) + ' schema:name ?d' + str(h + 1) + 'name.'

        def _generate_relationships(h: int, pred: str) -> str:
            relations = '\n    ?c1 ' + pred + ' ?c2.'
            for i in range(2, h + 1):
                relation = '\n     ?cc' + str(i) + ' ' + pred + ' ?c' + str(i + 1) + '.'
                relations += relation
            return relations

        def _generate_select(h: int) -> str:
            selects = '\n ?c1name ?t1name ?t1path ?d1name'
            for i in range(2, h + 1):
                selects += '\n ?c' + str(i) + 'name ?t' + str(i) + 'name ?t' + str(i) + 'path' \
                           ' ?cc' + str(i) + 'name ?d' + str(i) + 'name'
            selects += '\n ?c' + str(h+1) + 'name ?t' + str(h + 1) + 'name ?t' + str(h + 1) + 'path ?d'+str(h + 1)+'name'
            return selects
        starting_nodes = _generate_starting_nodes()
        intermediate_nodes = _generate_intermediate_nodes(hops)
        target_nodes = _generate_target_nodes(hops)
        relationships = _generate_relationships(hops, predicate)
        select = _generate_select(hops)
        all_nodes = starting_nodes + intermediate_nodes + target_nodes
        query = queryTemplates.get_paths_between_tables(select, start_iri, all_nodes, relationships, target_iri,
                                                        hops)
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return []
        for result in bindings:
            yield _create_path_row(result, hops)

    ##############

    def search(self, keyword, search_type, max_hits=15, show_query=False):
        def create_query():
            if search_type == KWType.KW_SEMANTIC:
                return queryTemplates.get_search_template('?cn', keyword, max_hits)
            elif search_type == KWType.KW_TABLE:
                return queryTemplates.get_search_template('?fn', keyword, max_hits)

        query = create_query()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in results["results"]["bindings"]:
            yield _create_simple_df_row(result)  # fix score later

    def approximate_search(self, keyword, search_type, max_hits=15, show_query=False):
        def create_query():
            if search_type == KWType.KW_SEMANTIC:
                return queryTemplates.get_approximate_search_template('?cn', keyword, max_hits)
            elif search_type == KWType.KW_TABLE:
                return queryTemplates.get_approximate_search_template('?fn', keyword, max_hits)

        query = create_query()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in results["results"]["bindings"]:
            yield _create_simple_df_row(result)

    def get_neighbors(self, h, relation, max_hits=15, show_query=False):
        def create_query():
            classIDs = _make_values_of_ids(h)
            return queryTemplates.get_related_neighbors_template(relation, classIDs, max_hits)

        query = create_query()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in results["results"]["bindings"]:
            yield _create_get_neighbor_df_row(result)  # fix score later

    """
    ##########
    ## path ##
    ##########
    """

    def get_path_between_nodes(self, a, b, relation, maxHops, show_query=False):
        def create_path_df_row(rel, res):
            resultDic = {}
            preDct = {'pre_nid': get_id_from_uri(res['predecessor']['value']),
                      'pre_db_name': res['dn1']['value'],
                      'pre_file_name': res['fn1']['value'],
                      'pre_column_name': res['cn1']['value'],
                      'pre_data_type': res['dt1']['value']}  # fix score later
            resultDic.update(preDct)
            outDct = {'target_nid': get_id_from_uri(res['out']['value']),
                      'target_db_name': res['dn2']['value'],
                      'target_file_name': res['fn2']['value'],
                      'target_column_name': res['cn2']['value'],
                      'target_data_type': res['dt2']['value']}  # fix score later
            resultDic.update(outDct)
            resultDic.update({rel.name + '_score': res['score']['value']})
            return resultDic

        def create_query():
            aID = _make_values_of_ids(a)
            bID = _make_values_of_ids(b)
            return queryTemplates.get_path_between_nodes_template(aID, bID, relation, maxHops)

        query = create_query()
        if show_query:
            _print_query(query)
        results = self.process_query(query, JSON)
        bindings = results["results"]["bindings"]
        if not bindings:
            return bindings
        for result in results["results"]["bindings"]:
            dic = create_path_df_row(relation, result)
            yield dic
