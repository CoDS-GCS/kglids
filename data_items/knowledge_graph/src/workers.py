# TODO: [Refactor] Where is the best place to have this file/functions?
import os
import random
import string
# TODO: [Refactor] Get rid of this hack
import sys
sys.path.append('../../../')

import numpy as np
from numpy.linalg import norm
from datasketch import MinHash
from sklearn.cluster import DBSCAN

from rdf_resource import RDFResource
from triplet import Triplet
from utils import generate_label


def column_metadata_worker(column_profiles,  ontology, triples_output_tmp_dir):
    

    # TODO: [Refactor] have the names of predicates read from global project ontology object
    triples = []

    for column_profile in column_profiles:

        column_node = RDFResource(column_profile.get_column_id(), ontology['kglidsResource'])
        table_node = RDFResource(column_profile.get_table_id(), ontology['kglidsResource'])
        col_label = generate_label(column_profile.get_column_name(), 'en')

        # membership
        triples.append(Triplet(column_node, RDFResource('isPartOf', ontology['kglids']), table_node))
        triples.append(Triplet(column_node, RDFResource('type', ontology['rdf']),
                               RDFResource('Column', ontology['kglids'])))
        # metadata
        triples.append(Triplet(column_node, RDFResource('hasDataType', ontology['kglidsData']),
                               RDFResource(column_profile.get_data_type())))
        triples.append(Triplet(column_node, RDFResource('name', ontology['schema']),
                               RDFResource(column_profile.get_column_name())))
        triples.append(Triplet(column_node, RDFResource('hasTotalValueCount', ontology['kglidsData']),
                               RDFResource(column_profile.get_total_values_count())))
        triples.append(Triplet(column_node, RDFResource('hasDistinctValueCount', ontology['kglidsData']),
                               RDFResource(column_profile.get_distinct_values_count())))
        triples.append(Triplet(column_node, RDFResource('hasMissingValueCount', ontology['kglidsData']),
                               RDFResource(column_profile.get_missing_values_count())))
        triples.append(Triplet(column_node, RDFResource('label', ontology['rdfs']), RDFResource(col_label)))

        if column_profile.is_numeric():
            triples.append(Triplet(column_node, RDFResource('hasMedianValue', ontology['kglidsData']),
                                   RDFResource(column_profile.get_median())))
            triples.append(Triplet(column_node, RDFResource('hasMaxValue', ontology['kglidsData']),
                                   RDFResource(column_profile.get_max_value())))
            triples.append(Triplet(column_node, RDFResource('hasMinValue', ontology['kglidsData']),
                                   RDFResource(column_profile.get_min_value())))

    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
    with open(os.path.join(triples_output_tmp_dir, filename), 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(f"{triple}\n")
    
    return []
    
        
def column_pair_similarity_worker(column_idx, column_profiles, ontology, triples_output_tmp_dir, 
                                  semantic_similarity_threshold, numerical_content_threshold,
                                  deep_embedding_content_threshold, inclusion_dependency_threshold,
                                  minhash_content_threshold, pkfk_threshold, word_embedding):

    # load the column profiles
    column1_profile = column_profiles[column_idx]
    similarity_triples = []
    for j in range(column_idx+1, len(column_profiles)):
        column2_profile = column_profiles[j]
        
        if column1_profile.get_data_type() != column2_profile.get_data_type():
            continue
        if column1_profile.get_table_id() == column2_profile.get_table_id():
            continue

        semantic_triples = _compute_semantic_similarity(column1_profile, column2_profile, ontology,
                                                        semantic_similarity_threshold,
                                                        word_embedding_model=word_embedding)
        content_triples = _compute_content_similarity(column1_profile, column2_profile, ontology,
                                                      numerical_content_threshold=numerical_content_threshold,
                                                      minhash_content_threshold=minhash_content_threshold)
        deep_content_triples = _compute_numerical_deep_content_similarity(column1_profile, column2_profile, ontology,
                                                                          deep_embedding_content_threshold)
        inclusion_triples = _compute_numerical_inclusion_dependency(column1_profile, column2_profile, ontology,
                                                                    inclusion_dependency_threshold)
        pkfk_triples = _compute_primary_key_foreign_key_similarity(column1_profile, column2_profile, ontology,
                                                                   pkfk_threshold,
                                                                   content_similarity_triples=content_triples,
                                                                   deep_content_similarity_triples=deep_content_triples,
                                                                   inclusion_dependency_triples=inclusion_triples)

        similarity_triples.extend(semantic_triples + content_triples + deep_content_triples + inclusion_triples + pkfk_triples)
    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
    with open(os.path.join(triples_output_tmp_dir, filename), 'w', encoding='utf-8') as f:
        for triple in similarity_triples:
            f.write(f"{triple}\n")
    
    return []
    

def _compute_semantic_similarity(column1_profile, column2_profile, ontology,
                                 semantic_similarity_threshold: float, word_embedding_model) -> list:
    # TODO: [Refactor] have the names of predicates read from global project ontology object

    column1_label = generate_label(column1_profile.get_column_name(), 'en').get_text()
    column2_label = generate_label(column2_profile.get_column_name(), 'en').get_text()
    
    score = word_embedding_model.get_distance_between_column_labels(column1_label, column2_label)
    semantic_similarity_triples = []
    if score >= semantic_similarity_threshold:
        semantic_similarity_triples.extend(_create_column_similarity_triples(column1_profile, column2_profile, 
                                                                             'hasSemanticSimilarity', score, ontology))
    return semantic_similarity_triples


def _compute_content_similarity(col1_profile, col2_profile, ontology,
                                numerical_content_threshold, minhash_content_threshold):
    # TODO: [Refactor] have the names of predicates read from global project ontology object
    content_similarity_triples = []
    if col1_profile.is_numeric():
        # TODO: [Implement] Remove non-deep similarities
        return []
        
        # Content similarity without deep embedding
        # TODO: [Implement] This original implementation generates duplicated similarity triples between columns with 
        #       different scores.
        """ 
         For example: 
        <<col1 hasContentSimilarity col2>> withCertainty 0.97
        <<col2 hasContentSimilarity col1>> withCertainty 0.97
        <<col1 hasContentSimilarity col2>> withCertainty 0.96
        <<col2 hasContentSimilarity col1>> withCertainty 0.96

        Proposed solution: make the scores asymmetric, i.e. have:
        <<col1 hasContentSimilarity col2>> withCertainty 0.97
        <<col2 hasContentSimilarity col1>> withCertainty 0.96
        """
        if col1_profile.get_iqr() != 0 and col2_profile.get_iqr() != 0:
            # columns with multiple unique values
            overlap1 = _compute_inclusion_overlap(col1_profile, col2_profile)
            overlap2 = _compute_inclusion_overlap(col2_profile, col1_profile)
            if overlap1 > numerical_content_threshold:
                content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                    'hasContentSimilarity', overlap1,
                                                                                    ontology))
            if overlap1 != overlap2 and overlap2 > numerical_content_threshold:
                content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                    'hasContentSimilarity', overlap2,
                                                                                    ontology))
            
        elif col1_profile.get_iqr() == 0 and col2_profile.get_iqr() == 0:
            # columns with single unique value
            # TODO: [Implement] I don't think DBSCAN is needed anymore
            medians = np.array([col1_profile.get_median() / 2, col2_profile.get_median() / 2]).reshape(-1, 1)
            predicted_clusters = DBSCAN(eps=0.1, min_samples=2).fit_predict(medians)
            if predicted_clusters[0] != -1:
                # i.e. the two medians are determined to be of the same cluster
                content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                    'hasContentSimilarity',
                                                                                    numerical_content_threshold, 
                                                                                    ontology))
                
    elif col1_profile.is_textual():
        column1_minhash = MinHash(num_perm=512, hashvalues=col1_profile.get_minhash())
        column2_minhash = MinHash(num_perm=512, hashvalues=col2_profile.get_minhash())
        similarity = column1_minhash.jaccard(column2_minhash)
        if similarity >= minhash_content_threshold:
            content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                'hasContentSimilarity', similarity,
                                                                                ontology))

    return content_similarity_triples


# TODO: [Refactor] combine this with content similarity method
def _compute_numerical_deep_content_similarity(col1_profile, col2_profile, ontology,
                                               deep_embedding_content_threshold):
    deep_content_similarity_triples = []
    if col1_profile.is_numeric():
        # Numerical column: calculate similarity with and without deep embeddings
        embedding_cos_sim = np.dot(col1_profile.get_deep_embedding(), col2_profile.get_deep_embedding()) / \
                            (norm(col1_profile.get_deep_embedding()) * norm(col2_profile.get_deep_embedding()))
        if embedding_cos_sim >= deep_embedding_content_threshold:
            deep_content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                     'hasContentSimilarity',
                                                                                     embedding_cos_sim,
                                                                                     ontology))
    return deep_content_similarity_triples


def _compute_numerical_inclusion_dependency(col1_profile, col2_profile, ontology, 
                                            inclusion_dependency_threshold):
    
    if not col1_profile.is_numeric():
        # inclusion dependency applies only for numerical columns
        return []
    if col1_profile.get_iqr() == 0 or col2_profile.get_iqr() == 0:
        return []
    if any([np.isinf(float(i)) for i in [col1_profile.get_min_value(), col1_profile.get_max_value(),
                                         col2_profile.get_min_value(), col2_profile.get_max_value()]]):
        return []

    # inclusion relation
    # TODO: [Implement] This original implementation generates duplicated similarity triples between columns with 
    #       different scores.
    """ 
    For example: 
    <<col1 hasInclusionDependency col2>> withCertainty 0.97
    <<col2 hasInclusionDependency col1>> withCertainty 0.97
    <<col1 hasInclusionDependency col2>> withCertainty 0.96
    <<col2 hasInclusionDependency col1>> withCertainty 0.96
    
    Proposed solution: make the scores asymmetric, i.e. have:
    <<col1 hasInclusionDependency col2>> withCertainty 0.97
    <<col2 hasInclusionDependency col1>> withCertainty 0.96
    """
    inclusion_dependency_triples = []
    if col2_profile.get_min_value() >= col1_profile.get_min_value() \
            and col2_profile.get_max_value() <= col1_profile.get_max_value():
        overlap = _compute_inclusion_overlap(col1_profile, col2_profile)
        if overlap >= inclusion_dependency_threshold:
            inclusion_dependency_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                  'hasInclusionDependency', overlap,
                                                                                  ontology))
    if col1_profile.get_min_value() >= col2_profile.get_min_value() \
            and col1_profile.get_max_value() <= col2_profile.get_max_value():
        overlap = _compute_inclusion_overlap(col2_profile, col1_profile)
        if overlap >= inclusion_dependency_threshold:
            inclusion_dependency_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                  'hasInclusionDependency', overlap,
                                                                                  ontology))
    return inclusion_dependency_triples


# TODO: [Refactor] this method needs to be moved somewhere else.
def _compute_inclusion_overlap(column1_profile, column2_profile):
    col1_left = column1_profile.get_median() - column1_profile.get_iqr()
    col1_right = column1_profile.get_median() + column1_profile.get_iqr()
    col2_left = column2_profile.get_median() - column2_profile.get_iqr()
    col2_right = column2_profile.get_median() + column2_profile.get_iqr()
    overlap = 0
    if col2_left >= col1_left and col2_right <= col1_right:
        overlap = float((col2_right - col2_left) / (col1_right - col1_left))
    elif col1_left <= col2_left <= col1_right:
        domain_ov = col1_right - col2_left
        overlap = float(domain_ov / (col1_right - col1_left))
    elif col1_left <= col2_right <= col1_right:
        domain_ov = col2_right - col1_left
        overlap = float(domain_ov / (col1_right - col1_left))
    return float(overlap)


def _compute_primary_key_foreign_key_similarity(col1_profile, col2_profile, ontology, 
                                                pkfk_threshold, content_similarity_triples, 
                                                deep_content_similarity_triples, inclusion_dependency_triples):
    # no pkfk if the columns are booleans of floats
    if col1_profile.is_boolean() or col1_profile.is_float():
        return []
    
    # we have pkfk if the two columns have content similarity and their cardinalities are above the provided threshold
    if col1_profile.get_total_values_count() == 0 or col2_profile.get_total_values_count() == 0:
        return []
    col1_cardinality = float(col1_profile.get_distinct_values_count()) / float(col1_profile.get_total_values_count())
    col2_cardinality = float(col2_profile.get_distinct_values_count()) / float(col2_profile.get_total_values_count())
    if min(col1_cardinality, col2_cardinality) < pkfk_threshold:
        return []
    
    highest_cardinality = max(col1_cardinality, col2_cardinality)
    pkfk_similarity_triples = []

    if col1_profile.is_numeric() and deep_content_similarity_triples:
        pkfk_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                         'hasPrimaryKeyForeignKeySimilarity',
                                                                         highest_cardinality, ontology))
    
    elif col1_profile.is_textual() and content_similarity_triples:
        pkfk_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile, 
                                                                         'hasPrimaryKeyForeignKeySimilarity', 
                                                                         highest_cardinality, ontology))


    return pkfk_similarity_triples


# TODO: [Refactor] this method needs to be moved somewhere else.
# TODO: [Refactor] this method shouldn't take ontology from the worker methods. It should be passed from global config
def _create_column_similarity_triples(column1_profile, column2_profile,
                                      similarity_predicate, similarity_score, ontology):
    nested_subject = RDFResource(column1_profile.get_column_id(), ontology['kglidsResource'])
    nested_predicate = RDFResource(similarity_predicate, ontology['kglidsData'])
    nested_object = RDFResource(column2_profile.get_column_id(), ontology['kglidsResource'])
    # RDF-star triple of the content similarity between the two columns with the score.
    triple = Triplet(Triplet(nested_subject, nested_predicate, nested_object),
                     RDFResource('withCertainty', ontology['kglidsData']), RDFResource(similarity_score))
    return [triple, triple.get_reversed_triple()]
