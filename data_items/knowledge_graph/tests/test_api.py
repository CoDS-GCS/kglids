import sys

sys.path.insert(0, '../src')
from api.api import API
from storage.elasticsearch_client import ElasticsearchClient
from storage.kglac_client import KGLacClient
from enums.relation import Relation
import pandas as pd

"""
##########
test the schema
##########
"""

store = ElasticsearchClient()
api = API(store, KGLacClient('filtered_v1'))
api2 = API(store, KGLacClient('la_demo3'))
api3 = API(store, KGLacClient('demo'))
api4 = API(store, KGLacClient('test_annotate'))
api5 = API(store, KGLacClient('demo_ahmed'))


def test_get_shortest_path_between_columns():
    #
    series1 = pd.Series({'column_name': 'writein', 'table_name': '1976-2020-president.csv',
                         'dataset_name': 'tunguz_us-elections-dataset'})

    series2 = pd.Series({'column_name': 'writein', 'table_name': '1976-2020-senate.csv',
                         'dataset_name': 'tunguz_us-elections-dataset'})

    result = api3.get_shortest_path_between_columns(series1, series2, via='semanticSimilarity', max_hops=2)
    assert (len(result) == 3)
    assert (len(result.columns) == 12)


def test_unionable_columns():
    df1 = pd.DataFrame([], columns=['Year', 'Employee Last Name', 'Employee First Name', 'Middle Initial'])

    df2 = pd.DataFrame([], columns=['Id', 'EmployeeName', 'JobTitle', 'BasePay', 'Year', 'Notes', 'Agency'])
    result = api2.get_unionable_columns(df1, df2)
    assert (result.shape == (3, 2))


def test_search_columns():
    gen_columns = api5.search_columns('gend')
    name_columns = api5.search_columns('name')
    assert (len(gen_columns) == 3)
    assert (len(name_columns) == 22)
    assert (len(name_columns.columns) == 12)


def test_search_tables():
    sal_columns = api5.search_tables('sala', show_query=True)
    class_columns = api5.search_tables('class')

    no_found_columns = api5.search_tables('hdfiwphfd[pw')
    assert (len(sal_columns) == 1)
    assert (len(class_columns) == 1)
    assert (len(no_found_columns) == 0)
    assert (len(no_found_columns.columns) == 6)
    assert (len(class_columns.columns) == 6)


def test_search_tables_on():
    gen_or_yee_all_and_ame_columns = api5.search_tables_on([['gend', 'sec'], ['ame']])
    nothing_columns = api5.search_tables_on([['nothing']])
    assert (len(gen_or_yee_all_and_ame_columns) == 1)
    assert (len(nothing_columns) == 0)
    assert (len(gen_or_yee_all_and_ame_columns.columns) == 6)


def test_joinable_columns_from_dataframe():
    empty_result = api2.get_joinable_columns(pd.DataFrame([]))
    assert (len(empty_result) == 0)

def test_joinable_columns_from_series():
    empty_result = api2.get_joinable_columns(pd.Series([]))
    assert(len(empty_result) == 0)


def test_joinable_columns_between():
    data1 = [('David', 1986, '1996'), ('Mary', 1996, '1796'), ('Justin', 1796, '1956')]
    data2 = [('David', 'dave', 12, ''), ('Mary', 'mary', 12, ''), ('Justin', 'justino', 11, '')]
    empty = []
    df1 = pd.DataFrame(data1, columns=['employee name', 'year', 'home #'])
    df2 = pd.DataFrame(data2, columns=['student name', 'age', 'nickanme', 'comments'])
    df3 = pd.DataFrame(empty, columns=['nothing', 'nothing', 'nothing'])
    joinable_on_name_df = api2.get_joinable_columns_between(df1, df2)
    none_joinable = api2.get_joinable_columns_between(df1, df3)
    assert (len(joinable_on_name_df) == 1)
    assert (len(joinable_on_name_df.columns) == 2)
    assert (len(none_joinable) == 0)


def test_number_of_dataset():
    number_of_datasets = api.get_number_of_datasets()
    assert (number_of_datasets['number_of_datasets'][0] == 1)


def test_get_all_tables():
    all_tables = api.get_all_tables()
    assert (len(all_tables) == 46)
    assert (sorted(all_tables.columns) == sorted(['db_name', 'column_name']))


def test_get_tables_in():
    tables_in_csv_repository = api.get_tables_in('csv_repository', max_results=100, show_query=True)
    tables_in_non_existing_dataset = api.get_tables_in('None')
    assert (len(tables_in_csv_repository) == 46)
    assert (sorted(tables_in_csv_repository.columns) == sorted(['db_name', 'column_name']))

    assert (len(tables_in_non_existing_dataset) == 0)
    assert (sorted(tables_in_non_existing_dataset) == sorted(['db_name', 'column_name']))


def test_search_approximate_table():
    single_keyword = 'site'
    compound_keyword = 'target_type'
    actual_result_compound_table_names = ['action_type.csv',
                                          'target_components.csv',
                                          'target_dictionary.csv',
                                          'target_relations.csv',
                                          'target_type.csv']

    single_result_df = api.search_table(single_keyword, approximate=True)
    compound_result_df = api.search_table(compound_keyword, approximate=True, max_results=100)

    result_single_table_names = single_result_df['file_name']

    result_compound_table_names = compound_result_df['file_name']

    assert (len(set(result_single_table_names)) == 2)
    assert (sorted(set(result_single_table_names)) == ['binding_sites.csv', 'site_components.csv'])

    assert (len(set(result_compound_table_names)) == 5)
    assert (sorted(set(result_compound_table_names)) == actual_result_compound_table_names)


def test_search_approximate_attribute():
    single_keyword = 'description'
    compound_keyword = 'src_description'
    actual_result_compound_table_names = ['action_type.csv', 'atc_classification.csv', 'cell_dictionary.csv',
                                          'component_sequences.csv', 'domains.csv', 'frac_classification.csv',
                                          'hrac_classification.csv', 'irac_classification.csv', 'source.csv']

    single_result_df = api.search_attribute(single_keyword, True, 100)
    compound_result_df = api.search_attribute(compound_keyword, True, 100)

    result_single_table_names = single_result_df['file_name']

    result_compound_table_names = compound_result_df['file_name']

    assert (len(set(result_single_table_names)) == 9)
    assert (sorted(set(result_single_table_names)) == actual_result_compound_table_names)

    assert (len(set(result_compound_table_names)) == 9)
    assert (sorted(set(result_compound_table_names)) == actual_result_compound_table_names)


def test_search_attribute():
    actual_ids = ['1543395099', '178231520', '2579025366', '3616735203', '3771643745']

    actual_source_name = ['binding_sites.csv', 'drug_mechanism.csv', 'target_components.csv', 'target_dictionary.csv',
                          'target_relations.csv']

    fn_attr = api.search_attribute('tid', max_results=30)
    doc_attr = api.search_attribute('doc_id')

    result_ids = fn_attr['nid']
    result_source_name = fn_attr['file_name']

    assert (len(doc_attr) == 1 and doc_attr['nid'][0] == '2096784535'
            and doc_attr['file_name'][0] == 'docs.csv')
    assert (len(fn_attr) == 5 and sorted(result_ids) == actual_ids
            and sorted(result_source_name) == actual_source_name)


def test_search_table():
    cell_dict_actual_ids = ['1163189569', '1320408914', '1702314070', '1928698180', '2841701016',
                            '3012738100', '3974301053', '898175268', '899909579', '91405611', '951481135']

    cell_dict_actual_field_names = ['cell_description', 'cell_id', 'cell_name', 'cell_source_organism',
                                    'cell_source_tax_id', 'cell_source_tissue', 'cellosaurus_id', 'chembl_id',
                                    'cl_lincs_id', 'clo_id', 'efo_id']

    lookup_actual_ids = ['1795451256', '1903343116', '2541662707', '374920930']
    lookup_actual_field_names = ['chembl_id', 'entity_id', 'entity_type', 'status']

    cell_dict_table = api.search_table('cell_dictionary.csv', max_results=100)
    lookup_table = api.search_table('chembl_id_lookup.csv', max_results=100)

    cell_dict_result_ids = cell_dict_table['nid']
    cell_dict_result_column_name = cell_dict_table['column_name']

    lookup_result_ids = lookup_table['nid']
    lookup_result_column_name = lookup_table['column_name']

    # Assert the cell_dictionary.csv properties
    assert (len(cell_dict_table) == 11)
    assert (sorted(cell_dict_result_ids) == cell_dict_actual_ids)
    assert (sorted(cell_dict_result_column_name) == cell_dict_actual_field_names)

    # Assert the chembl_id_lookup.csv properties
    assert (len(lookup_table) == 4)
    assert (sorted(lookup_result_ids) == lookup_actual_ids)
    assert (sorted(lookup_result_column_name) == lookup_actual_field_names)


def test_pkfk():
    # test a field:
    result_pkfks_doc_id = api.get_pkfk_of('2096784535', show_query=True)

    result_doc_id_ids = result_pkfks_doc_id['nid']
    result_doc_id_field_names = result_pkfks_doc_id['column_name']
    result_doc_id_source_names = result_pkfks_doc_id['file_name']

    assert (len(result_pkfks_doc_id) == 2)
    assert (sorted(result_doc_id_field_names) == ['drugind_id', 'drugind_id'])
    assert (sorted(result_doc_id_ids) == ['1609351779', '1673532872'])
    assert (sorted(result_doc_id_source_names) == ['drug_indication.csv', 'indication_refs.csv'])

    # test a table:
    doc_table_hit_list = \
        [{'nid': '1349017573', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'last_page',
          'score': 4.4385214},
         {'nid': '2439808217', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'doc_type',
          'score': 4.4385214},
         {'nid': '2974113641', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'issue',
          'score': 4.3892713},
         {'nid': '574979901', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'authors',
          'score': 4.3892713},
         {'nid': '4026421333', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'year',
          'score': 4.337475},
         {'nid': '1278665205', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'volume',
          'score': 4.337475},
         {'nid': '3303038860', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'doi',
          'score': 4.019371},
         {'nid': '2296251452', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'title',
          'score': 4.019371},
         {'nid': '832387975', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'abstract',
          'score': 4.019371},
         {'nid': '2096784535', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'doc_id',
          'score': 3.269366},
         {'nid': '1844359713', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'journal',
          'score': 3.269366},
         {'nid': '1794730969', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'first_page',
          'score': 3.269366},
         {'nid': '4102463509', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'pubmed_id',
          'score': 3.269366},
         {'nid': '877015082', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'chembl_id',
          'score': 3.269366},
         {'nid': '3286487763', 'db_name': 'csv_repository', 'source_name': 'docs.csv', 'field_name': 'patent_id',
          'score': 3.269366}]

    actual_doc_table_pkfk_ids = ['1028710623', '1041698491', '1609351779',
                                 '1673532872', '1685870800', '1773829061',
                                 '1820036852', '1974922703', '2090975805',
                                 '2196498330', '2250661389', '2295354561',
                                 '2753114679', '2801995080', '298809914',
                                 '3047147543', '3123832564',
                                 '3462460652', '3699272886', '4028189684', '422903557',
                                 '4252615943', '77108614', '879755307']

    doc_df = pd.DataFrame(doc_table_hit_list)
    pkfk_of_doc_table_result = api.get_pkfk_of(doc_df, 50, show_query=True)

    result_doc_table_pkfk_ids = pkfk_of_doc_table_result['nid']
    assert (len(result_doc_table_pkfk_ids) == 24)
    print(sorted(result_doc_table_pkfk_ids))
    assert (sorted(result_doc_table_pkfk_ids) == actual_doc_table_pkfk_ids)


def test_content_similarity():
    # test a field:
    result_content_similarity_oc_id = api.get_content_similar_to('2196498330')
    actual_oc_id_ids = ['1041698491', '1291380165', '77108614', '841963878']
    actual_oc_id_field_names = ['comp_class_id', 'component_id', 'component_id',
                                'targcomp_id']

    result_oc_id_ids = result_content_similarity_oc_id['nid']
    result_oc_id_field_names = result_content_similarity_oc_id['column_name']

    assert (len(result_content_similarity_oc_id) == 4)
    assert (sorted(result_oc_id_field_names) == actual_oc_id_field_names)
    assert (sorted(result_oc_id_ids) == actual_oc_id_ids)

    # test a table:
    class_organism_table_hit_list = [
        {'nid': '871959201', 'db_name': 'csv_repository', 'source_name': 'organism_class.csv', 'field_name': 'tax_id',
         'score': 7.4439383},
        {'nid': '3360543807', 'db_name': 'csv_repository', 'source_name': 'organism_class.csv', 'column_name': 'l1',
         'score': 6.7180624},
        {'nid': '1363445125', 'db_name': 'csv_repository', 'source_name': 'organism_class.csv', 'column_name': 'l2',
         'score': 6.7180624},
        {'nid': '2196498330', 'db_name': 'csv_repository', 'source_name': 'organism_class.csv', 'column_name': 'oc_id',
         'score': 6.5337486},
        {'nid': '641971475', 'db_name': 'csv_repository', 'source_name': 'organism_class.csv', 'column_name': 'l3',
         'score': 6.5337486}]

    actual_class_organism_table_content_similarity_ids = ['1041698491', '1291380165',
                                                          '478501033', '77108614', '841963878']

    classification_organism_df = pd.DataFrame(class_organism_table_hit_list)

    similar_content_of_classification_organism_table_result = api.get_content_similar_to(classification_organism_df, 50)

    result_class_organism_table_content_similarity_ids = similar_content_of_classification_organism_table_result['nid']

    assert (len(similar_content_of_classification_organism_table_result) == 5)
    assert (sorted(
        result_class_organism_table_content_similarity_ids) == actual_class_organism_table_content_similarity_ids)


def test_semantic_similarity():
    # test a field:
    result_semantic_similarity_description = api.get_semantically_similar_to('3557068642')
    actual_description_ids = ['3595621057']
    actual_description_field_names = ['description']

    result_description_ids = result_semantic_similarity_description['nid']
    result_description_field_names = result_semantic_similarity_description['column_name']

    assert (len(result_semantic_similarity_description) == 1)
    assert (sorted(result_description_field_names) == actual_description_field_names)
    assert (sorted(result_description_ids) == actual_description_ids)

    # test a table:
    action_type_table_hit_list = [
        {'nid': '1138697059', 'db_name': 'csv_repository', 'source_name': 'action_type.csv',
         'field_name': 'action_type',
         'score': 7.656536},
        {'nid': '148745945', 'db_name': 'csv_repository', 'source_name': 'action_type.csv', 'field_name': 'parent_type',
         'score': 7.551405},
        {'nid': '3557068642', 'db_name': 'csv_repository', 'source_name': 'action_type.csv',
         'field_name': 'description',
         'score': 6.8499203}]

    actual_action_type_table_schema_similarity_ids = ['1869348300', '3220141534', '3595621057']

    action_type_df = pd.DataFrame(action_type_table_hit_list)

    similar_schema_of_action_type_table_result = api.get_semantically_similar_to(action_type_df, 50, show_query=True)

    result_action_type_table_content_similarity_ids = similar_schema_of_action_type_table_result['nid']

    assert (len(similar_schema_of_action_type_table_result) == 3)
    assert (sorted(
        result_action_type_table_content_similarity_ids) == actual_action_type_table_schema_similarity_ids)


def test_intersection_field_mode():
    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    doc_version_intersection_df = api.get_intersection_between(doc_df, version_df)
    name_version_intersection_df = api.get_intersection_between(version_df, name_df)

    assert (len(doc_version_intersection_df) == 0)
    assert (len(name_version_intersection_df) == 1)
    assert (name_version_intersection_df['nid'][0] == '1719221725')
    assert (list(name_version_intersection_df.loc[0]) == list(name_df.loc[0]))


def test_intersection_table_mode():
    actual_ids = ['1719221725', '1719221725', '1719221725', '1719221725', '3024197227', '690247821']
    result_ids = []

    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    doc_version_intersection_df = api.get_intersection_between(doc_df, version_df, True)
    name_version_intersection_df = api.get_intersection_between(version_df, name_df, True)

    result_ids.extend(list(name_version_intersection_df['nid_1']))
    result_ids.extend(list(name_version_intersection_df['nid_2']))

    assert (len(doc_version_intersection_df) == 0)
    assert (len(result_ids) == 6)
    assert (sorted(result_ids) == actual_ids)


def test_union_field_mode():
    actual_doc_version_ids = ['1278665205', '1349017573', '1719221725', '1794730969',
                              '1844359713', '2096784535', '2296251452', '2439808217',
                              '2974113641', '3024197227', '3286487763', '3303038860',
                              '4026421333', '4102463509', '574979901', '690247821',
                              '832387975', '877015082']
    actual_name_version_ids = ['1719221725', '3024197227', '690247821']

    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    doc_version_union_df = api.get_union_between(doc_df, version_df)
    name_version_union_df = api.get_union_between(version_df, name_df)

    result_doc_version_ids = doc_version_union_df['nid']
    result_name_version_ids = name_version_union_df['nid']

    assert (len(doc_version_union_df) == 18)
    assert (sorted(result_doc_version_ids) == actual_doc_version_ids)

    assert (len(name_version_union_df) == 3)
    assert (sorted(result_name_version_ids) == actual_name_version_ids)


def test_union_table_mode():
    actual_doc_version_ids = ['1278665205', '1349017573', '1719221725', '1794730969',
                              '1844359713', '2096784535', '2296251452', '2439808217',
                              '2974113641', '3024197227', '3286487763', '3303038860',
                              '4026421333', '4102463509', '574979901', '690247821',
                              '832387975', '877015082']
    actual_name_version_ids = ['1719221725', '3024197227', '690247821']

    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    doc_version_union_df = api.get_union_between(doc_df, version_df)
    name_version_union_df = api.get_union_between(version_df, name_df)

    result_doc_version_ids = doc_version_union_df['nid']
    result_name_version_ids = name_version_union_df['nid']

    assert (len(doc_version_union_df) == 18)
    assert (sorted(result_doc_version_ids) == actual_doc_version_ids)

    assert (len(name_version_union_df) == 3)
    assert (sorted(result_name_version_ids) == actual_name_version_ids)


def test_difference_field_mode():
    actual_version_docs_ids = ['1719221725', '3024197227', '690247821']
    actual_version_name_ids = ['3024197227', '690247821']

    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    version_doc_difference_df = api.get_difference_between(version_df, doc_df)
    version_name_difference_df = api.get_difference_between(version_df, name_df)

    result_version_docs_ids = version_doc_difference_df['nid']
    result_version_name_ids = version_name_difference_df['nid']

    assert (len(version_doc_difference_df) == 3)
    assert (sorted(result_version_docs_ids) == actual_version_docs_ids)

    assert (len(version_name_difference_df) == 2)
    assert (sorted(result_version_name_ids) == actual_version_name_ids)


def test_difference_table_mode():
    actual_version_docs_ids = ['1719221725', '3024197227', '690247821']
    actual_version_name_ids = ['3024197227', '690247821']

    doc_df = api.search_table('docs.csv')
    version_df = api.search_table('version.csv')
    name_df = api.search_attribute('name')  # name is an attribute in version.csv

    version_doc_difference_df = api.get_difference_between(version_df, doc_df)
    version_name_difference_df = api.get_difference_between(version_df, name_df)

    result_version_docs_ids = version_doc_difference_df['nid']
    result_version_name_ids = version_name_difference_df['nid']

    assert (len(version_doc_difference_df) == 3)
    assert (sorted(result_version_docs_ids) == actual_version_docs_ids)

    assert (len(version_name_difference_df) == 2)
    assert (sorted(result_version_name_ids) == actual_version_name_ids)


def test_path_between_hits():
    a = {'nid': '2579025366', 'db_name': 'csv_repository', 'file_name': 'target_relations.csv', 'column_name': 'tid',
         'score': 4.0943446}
    b = {'nid': '178231520', 'db_name': 'csv_repository', 'file_name': 'drug_mechanism.csv',
         'column_name': 'tid', 'score': 4.0943446}

    found_df = api.get_path_between_nodes(a, b, Relation.pkfk, 2, show_query=True)
    not_found_df = api.get_path_between_nodes(a, b, Relation.pkfk, 1, show_query=True)
    ids = [tuple(x) for x in found_df[['pre_nid', 'target_nid']].to_numpy()]

    assert (ids == [('2579025366', '3771643745'), ('3771643745', '178231520')])
    assert (len(found_df) == 2)
    assert (len(not_found_df) == 0)


def test_paths():
    drs1Data = [{'nid': '2579025366', 'db_name': 'csv_repository', 'file_name': 'target_relations.csv',
                 'column_name': 'tid', 'score': 4.0943446},
                {'nid': '3771643745', 'db_name': 'csv_repository', 'file_name': 'binding_sites.csv',
                 'column_name': 'tid', 'score': 4.0943446}]
    drs2Data = [{'nid': '178231520', 'db_name': 'csv_repository', 'file_name': 'drug_mechanism.csv',
                 'column_name': 'tid', 'score': 4.0943446}]

    df1 = pd.DataFrame(drs1Data)
    df2 = pd.DataFrame(drs2Data)

    paths1_2 = api.get_path_between_dfs(df1, df2)
    paths1_1 = api.get_path_between_dfs(df1, None)
    ids1_2 = []
    ids1_1 = []
    for i in range(paths1_2.index.levshape[0]):
        ids1_2.append([tuple(x) for x in paths1_2.loc[[i]][['pre_nid', 'target_nid']].to_numpy()])

    for i in range(paths1_1.index.levshape[0]):
        ids1_1.append([tuple(x) for x in paths1_1.loc[[i]][['pre_nid', 'target_nid']].to_numpy()])

    assert (len(ids1_2) == 2)
    assert (ids1_2 == [[('2579025366', '3771643745'), ('3771643745', '178231520')],
                       [('3771643745', '178231520')]])

    assert (len(ids1_1) == 1)
    assert (ids1_1 == [[('2579025366', '3771643745')]])
