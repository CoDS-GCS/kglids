import sys
from collections import Counter

sys.path.insert(0, '../src')
from rdf_builder import RDFBuilder

data = [('1', 'test', 'test_db', 'test1_tn', 'test_cn', 10, 5, 2, 'N', 5, 0, 10, 'path1'),
        ('2', 'test', 'test_db', 'test2_tn', 'test_cn', 10, 10, 5, 'T', -1, -1, -1, 'path2'),
        ('3', 'test', 'test_db', 'test1_tn', 'test_cn', 0, 0, 10, 'N', 0, 0, 0, 'path1')]


def test_initialize_nodes():
    rdf_builder = RDFBuilder()
    rdf_builder.initialize_nodes(data)
    table_id_to_name = rdf_builder._table_id_to_name
    dataset_id_to_name = rdf_builder._dataset_id_to_name
    result_triplets = rdf_builder.get_triplets()
    test1_node = '<http://www.example.com/lac#' + table_id_to_name['test1_tn'] + '>'
    test2_node = '<http://www.example.com/lac#' + table_id_to_name['test2_tn'] + '>'
    db1_node = '<http://www.example.com/lac#' + dataset_id_to_name['test_db'] + '>'

    actual_triplets = ['<http://www.example.com/lac#1> <http://schema.org/type> \"N\".',
                       '<http://www.example.com/lac#1> <http://www.w3.org/2002/07/owl#cardinality> 0.5.',
                       '<http://www.example.com/lac#1> <http://purl.org/dc/terms/isPartOf> ' + test1_node + '.',
                       '<http://www.example.com/lac#1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                       ' <http://www.example.com/lac#column>.',
                       '<http://www.example.com/lac#1> <http://schema.org/name> \"test_cn\".',
                       '<http://www.example.com/lac#1> <http://schema.org/totalVCount> 10.',
                       '<http://www.example.com/lac#1> <http://schema.org/distinctVCount> 5.',
                       '<http://www.example.com/lac#1> <http://schema.org/missingVCount> 2.',
                       '<http://www.example.com/lac#1> <http://schema.org/median> 5.',
                       '<http://www.example.com/lac#1> <http://schema.org/maxValue> 10.',
                       '<http://www.example.com/lac#1> <http://schema.org/minValue> 0.',
                       '<http://www.example.com/lac#1> <http://www.example.com/lac#origin> \"test\".',
                       '<http://www.example.com/lac#1> <http://www.w3.org/2000/01/rdf-schema#label> \"test cn\"@en.',

                       '<http://www.example.com/lac#2> <http://schema.org/type> \"T\".',
                       '<http://www.example.com/lac#2> <http://www.w3.org/2002/07/owl#cardinality> 1.0.',
                       '<http://www.example.com/lac#2> <http://purl.org/dc/terms/isPartOf> ' + test2_node + '.',
                       '<http://www.example.com/lac#2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                       ' <http://www.example.com/lac#column>.',
                       '<http://www.example.com/lac#2> <http://schema.org/name> \"test_cn\".',
                       '<http://www.example.com/lac#2> <http://schema.org/totalVCount> 10.',
                       '<http://www.example.com/lac#2> <http://schema.org/distinctVCount> 10.',
                       '<http://www.example.com/lac#2> <http://schema.org/missingVCount> 5.',
                       '<http://www.example.com/lac#2> <http://www.example.com/lac#origin> \"test\".',
                       '<http://www.example.com/lac#2> <http://www.w3.org/2000/01/rdf-schema#label> \"test cn\"@en.',

                       '<http://www.example.com/lac#3> <http://schema.org/type> \"N\".',
                       '<http://www.example.com/lac#3> <http://purl.org/dc/terms/isPartOf> ' + test1_node + '.',
                       '<http://www.example.com/lac#3> <http://schema.org/name> \"test_cn\".',
                       '<http://www.example.com/lac#3> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                       ' <http://www.example.com/lac#column>.',
                       '<http://www.example.com/lac#3> <http://schema.org/totalVCount> 0.',
                       '<http://www.example.com/lac#3> <http://schema.org/distinctVCount> 0.',
                       '<http://www.example.com/lac#3> <http://schema.org/missingVCount> 10.',
                       '<http://www.example.com/lac#3> <http://schema.org/median> 0.',
                       '<http://www.example.com/lac#3> <http://schema.org/maxValue> 0.',
                       '<http://www.example.com/lac#3> <http://schema.org/minValue> 0.',
                       '<http://www.example.com/lac#3> <http://www.example.com/lac#origin> \"test\".',
                       '<http://www.example.com/lac#3> <http://www.w3.org/2000/01/rdf-schema#label> \"test cn\"@en.',

                       test1_node + ' <http://schema.org/name> \"test1_tn\".',
                       test1_node + ' <http://purl.org/dc/terms/isPartOf> ' + db1_node + '.',
                       test1_node + ' <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                                    ' <http://www.example.com/lac#table>.',
                       test1_node + ' <http://www.w3.org/2000/01/rdf-schema#label> \"test 1 tn\"@en.',
                       test1_node + ' <http://www.example.com/lac#path> \"path1\".',

                       test2_node + ' <http://schema.org/name> \"test2_tn\".',
                       test2_node + ' <http://purl.org/dc/terms/isPartOf> ' + db1_node + '.',
                       test2_node + ' <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                                    ' <http://www.example.com/lac#table>.',
                       test2_node + ' <http://www.w3.org/2000/01/rdf-schema#label> \"test 2 tn\"@en.',
                       test2_node + ' <http://www.example.com/lac#path> \"path2\".',

                       db1_node + ' <http://schema.org/name> \"test_db\".',
                       db1_node + ' <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
                                  ' <http://www.example.com/lac#dataset>.',
                       db1_node + ' <http://www.w3.org/2000/01/rdf-schema#label> \"test db\"@en.',

                       test1_node + ' <http://www.example.com/lac#origin> \"test\".',
                       test2_node + ' <http://www.example.com/lac#origin> \"test\".'
                       ]

    assert (Counter(result_triplets) == Counter(actual_triplets))


def test_create_semantic_sim_relation():
    rdf_builder = RDFBuilder()
    rdf_builder.initialize_nodes(data)
    rdf_builder.build_semantic_sim_relation()
    result_triplets = rdf_builder.get_triplets()
    actual_triplets = ['<http://www.example.com/lac#1> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#3>.',

                       '<http://www.example.com/lac#1> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#2>.',

                       '<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#3>.',

                       '<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#2>.',

                       '<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#1>.',

                       '<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#1>.',

                       '<<<http://www.example.com/lac#1> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#3>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#1>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',

                       '<<<http://www.example.com/lac#1> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#1>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',

                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#3>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#semanticSimilarity> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.'
                       ]
    assert (Counter(actual_triplets) == Counter(result_triplets))


def test_create_txt_content_similarity():
    mh_signatures = [('1', [i % 2 for i in range(512)]), ('2', [i % 5 for i in range(512)]),
                     ('3', [i % 3 for i in range(512)]), ('4', [i % 5 for i in range(512)])]
    txt_content_data = [('1', 'test', 'test_db', 'test1_tn', 'test1_cn', 10, 5, 2, 'N', 5, 0, 10, 'path1'),
                        ('2', 'test', 'test_db', 'test2_tn', 'test_cn', 10, 10, 0, 'T', -1, -1, -1, 'path2'),
                        ('3', 'test', 'test_db', 'test1_tn', 'test3_cn', 0, 0, 5, 'N', 0, 0, 0, 'path1'),
                        ('4', 'test', 'test_db', 'test2_tn', 'test_cn', 0, 0, 3, 'N', 0, 0, 0, 'path2')]
    rdf_builder = RDFBuilder()
    rdf_builder.initialize_nodes(txt_content_data)
    rdf_builder.build_semantic_sim_relation()
    rdf_builder.build_content_sim_mh_text(mh_signatures)
    actual_triplets = ['<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#4>.',

                       '<http://www.example.com/lac#4> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#2>.',

                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#4>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#4> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.']
    result_triplets = rdf_builder.get_triplets()
    rdf_builder.build_content_sim_mh_text(mh_signatures[:2])
    assert (Counter(actual_triplets) == Counter(result_triplets))
    assert ([] == rdf_builder.get_triplets())


def test_create_num_content_similarity():
    id_signatures = [('1', (5, 4, 100, 120)), ('2', (3, 1, 2, 5)),
                     ('3', (3, 1, 3, 4))]

    rdf_builder = RDFBuilder()
    rdf_builder.initialize_nodes(data)
    rdf_builder.build_semantic_sim_relation()
    rdf_builder.build_content_sim_relation_num_overlap_distr(id_signatures)
    result_triplets = rdf_builder.get_triplets()
    actual_triplets = ['<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#2>.',

                       '<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#3>.',

                       '<<<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#contentSimilarity> '
                       '<http://www.example.com/lac#3>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',

                       '<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#inclusionDependency> '
                       '<http://www.example.com/lac#2>.',

                       '<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#inclusionDependency> '
                       '<http://www.example.com/lac#3>.',

                       '<<<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#inclusionDependency> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#inclusionDependency> '
                       '<http://www.example.com/lac#3>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.'
                       ]
    assert (Counter(actual_triplets) == Counter(result_triplets))


def test_create_pkfk_relations():
    pkfk_data = [('1', 'test', 'test_db', 'test_tn', 'test1_cn', 0, 0, 0, 'N', 0, 0, 0, 'path'),
                 ('2', 'test', 'test_db', 'test_tn', 'test2_cn', 10, 10, 0, 'T', -1, -1, -1, 'path'),
                 ('3', 'test', 'test_db', 'test_tn', 'test3_cn', 10, 8, 0, 'N', 4, 0, 10, 'path'),
                 ('4', 'test', 'test_db', 'test_tn', 'test4_cn', 10, 9, 0, 'N', 6, 2, 10, 'path'),
                 ('5', 'test', 'test_db', 'test_tn', 'test5_cn', 14, 3, 2, 'T', -1, -1, -1, 'path'),
                 ('6', 'test', 'test_db', 'test_tn', 'test6_cn', 11, 9, 1, 'T', -1, -1, -1, 'path')]

    num_signatures = [('3', (3, 1, 2, 5)), ('1', (105, 1, 100, 110)),
                      ('4', (3, 1, 3, 4))]

    txt_signatures = [('2', [i % 2 for i in range(512)]), ('5', [i % 5 for i in range(512)]),
                      ('6', [i % 2 for i in range(512)])]

    rdf_builder = RDFBuilder()
    rdf_builder.initialize_nodes(pkfk_data)
    rdf_builder.build_semantic_sim_relation()
    rdf_builder.build_content_sim_mh_text(txt_signatures)
    rdf_builder.build_content_sim_relation_num_overlap_distr(num_signatures)
    rdf_builder.build_pkfk_relation()
    result_triplets = rdf_builder.get_triplets()

    actual_triplets = ['<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#6>.',

                       '<http://www.example.com/lac#6> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#2>.',

                       '<<<http://www.example.com/lac#2> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#6>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',
                       '<<<http://www.example.com/lac#6> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#2>>> '
                       '<http://www.example.com/lac#certainty> '
                       '1.0.',

                       '<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#4>.',

                       '<http://www.example.com/lac#4> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#3>.',

                       '<<<http://www.example.com/lac#3> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#4>>> '
                       '<http://www.example.com/lac#certainty> '
                       '0.9.',
                       '<<<http://www.example.com/lac#4> '
                       '<http://www.example.com/lac#pkfk> '
                       '<http://www.example.com/lac#3>>> '
                       '<http://www.example.com/lac#certainty> '
                       '0.9.']
    assert (Counter(actual_triplets) == Counter(result_triplets))
