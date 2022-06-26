import os
import re
import json
import random
import string
import shutil
from datetime import datetime
import random
import string
import sys
import argparse
import pickle
import itertools

sys.path.append('../../../')

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

import numpy as np
from numpy.linalg import norm
from datasketch import MinHash
from sklearn.cluster import DBSCAN
from camelsplit import camelsplit


# TODO: [Refactor] project structure needs to be changed. This import won't work in terminal without the above sys call.
from data_items.profiler.src.data.column_profile import ColumnProfile

# ************* SYSTEM PARAMETERS**********************
# TODO: [Refactor] have these inside a global project config
SEMANTIC_THRESHOLD = 0.50
CONTENT_THRESHOLD = 0.95
MINHASH_THRESHOLD = 0.70
INCLUSION_THRESHOLD = 0.90
DEEP_EMBEDDING_THRESHOLD = 0.95
PKFK_THRESHOLD = 0.60


# *****************************************************


class KnowledgeGraphBuilder:
    # TODO: [Refactor] add all kglids URIs to knowledge graph config.py
    # TODO: [Refactor] have Spark configuration read from the global project config
    # TODO: [Refactor] read raw word embeddings path from global project config
    def __init__(self, column_profiles_path, out_graph_path, spark_mode, memory_size):
        self.graph_output_path = out_graph_path
        self.out_graph_base_dir = os.path.dirname(self.graph_output_path)
        if os.path.exists(self.graph_output_path):
            renamed_graph = f'{self.out_graph_base_dir}/OLD_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{self.graph_output_path.split("/")[-1]}'
            print(f'Found existing graph at: {self.graph_output_path}. Renaming to: {renamed_graph}')
            os.rename(self.graph_output_path, renamed_graph)
        self.tmp_graph_base_dir = os.path.join(self.out_graph_base_dir, 'tmp')  # for intermediate results
        if os.path.exists(self.tmp_graph_base_dir):
            shutil.rmtree(self.tmp_graph_base_dir)
        os.makedirs(self.tmp_graph_base_dir)

        self.memory_size = memory_size
        
        if spark_mode == 'cluster':
            self.spark = (SparkSession.builder
                                      .appName("KGBuilder") 
                                      .getOrCreate()
                                      .sparkContext)
        else:
            self.spark = SparkContext(conf=SparkConf().setMaster(f'local[*]')
                                                      .set('spark.driver.memory', f'{self.memory_size}g'))

        self.ontology = {'kglids': 'http://kglids.org/ontology/',
                         'kglidsData': 'http://kglids.org/ontology/data/',
                         'kglidsResource': 'http://kglids.org/resource/',
                         'schema': 'http://schema.org/',
                         'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                         'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'}

        # get list of column profiles and their data type
        column_data_types = [i for i in os.listdir(column_profiles_path)]
        self.column_profile_paths = []
        print('Column Type Breakdown:')
        for data_type in column_data_types:
            type_path = os.path.join(column_profiles_path, data_type)
            profiles = [os.path.join(type_path, i) for i in os.listdir(type_path) if i.endswith('.json')]
            self.column_profile_paths.extend(profiles)
            print(f'\t{data_type}: {len(profiles)}')
        print('Total:', len(self.column_profile_paths))
        
        # TODO: [Refactor] Read this from project config
        self.word_embedding_path = '../../data/glove.6B.100d.txt'
        # make sure the word embeddings are initialized
        self.word_embedding = WordEmbeddings(self.word_embedding_path)


    def build_membership_and_metadata_subgraph(self):
        
        # generate column metadata triples in parallel
        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        # mapPartitions so we don't end up with too many subgraph files (compared to .map())
        column_profile_paths_rdd = self.spark.parallelize(self.column_profile_paths)
        column_profile_paths_rdd.mapPartitions(lambda x: column_metadata_worker(column_profile_paths=x,
                                                                                ontology=ontology,
                                                                                triples_output_tmp_dir=tmp_graph_dir))\
                                .collect()
        # generate table and dataset membership triples
        membership_triples = []
        tables = set()
        datasets = set()
        sources = set()
        for column_profile_path in self.column_profile_paths:
            profile = ColumnProfile.load_profile(column_profile_path)
            if profile.get_table_id() in tables:
                continue
            # table -> dataset membership and metadata
            tables.add(profile.get_table_id())
            table_node = RDFResource(profile.get_table_id(), self.ontology['kglidsResource'])
            dataset_node = RDFResource(profile.get_dataset_id(), self.ontology['kglidsResource'])
            table_label = generate_label(profile.get_table_name(), 'en')
            membership_triples.append(Triplet(table_node, RDFResource('isPartOf', self.ontology['kglids']),
                                              RDFResource(dataset_node)))
            membership_triples.append(Triplet(table_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(profile.get_table_name())))
            membership_triples.append(Triplet(table_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(table_label)))
            membership_triples.append(Triplet(table_node, RDFResource('hasFilePath', self.ontology['kglidsData']),
                                              RDFResource(profile.get_path())))
            membership_triples.append(Triplet(table_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Table', self.ontology['kglids'])))

            if profile.get_dataset_id() in datasets:
                continue
            # dataset -> source membership and metadata
            datasets.add(profile.get_dataset_id())
            source_node = RDFResource(profile.get_datasource(), self.ontology['kglidsResource'])
            membership_triples.append(Triplet(dataset_node, RDFResource('isPartOf', self.ontology['kglids']),
                                              RDFResource(source_node)))
            dataset_label = generate_label(profile.get_dataset_name(), 'en')
            membership_triples.append(Triplet(dataset_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(profile.get_dataset_name())))
            membership_triples.append(Triplet(dataset_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(dataset_label)))
            membership_triples.append(Triplet(dataset_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Dataset', self.ontology['kglids'], False)))

            if profile.get_datasource() in sources:
                continue
            # source metadata
            sources.add(profile.get_datasource())
            source_label = generate_label(profile.get_datasource(), 'en')
            membership_triples.append(Triplet(source_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(profile.get_datasource())))
            membership_triples.append(Triplet(source_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(source_label)))
            membership_triples.append(Triplet(source_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Source', self.ontology['kglids'], False)))
        filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
        with open(os.path.join(self.tmp_graph_base_dir, filename), 'w', encoding='utf-8') as f:
            for triple in membership_triples:
                f.write(f"{triple}\n")

    def generate_similarity_triples(self):
        
        column_profile_paths = self.column_profile_paths
        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        word_embedding = self.word_embedding
        column_profile_indexes_rdd = self.spark.parallelize(list(range(len(self.column_profile_paths))))
        column_profile_indexes_rdd.map(
            lambda x: column_pair_similarity_worker(column_idx=x,
                                                    column_profile_paths=column_profile_paths,
                                                    ontology=ontology,
                                                    triples_output_tmp_dir=tmp_graph_dir,
                                                    semantic_similarity_threshold=SEMANTIC_THRESHOLD,
                                                    numerical_content_threshold=CONTENT_THRESHOLD,
                                                    deep_embedding_content_threshold=DEEP_EMBEDDING_THRESHOLD,
                                                    inclusion_dependency_threshold=INCLUSION_THRESHOLD,
                                                    minhash_content_threshold=MINHASH_THRESHOLD,
                                                    pkfk_threshold=PKFK_THRESHOLD,
                                                    word_embedding=word_embedding)) \
                                .collect()

    def build_graph(self):
        for tmp_file in os.listdir(self.tmp_graph_base_dir):
            with open(os.path.join(self.tmp_graph_base_dir, tmp_file), 'r') as f:
                content = f.read()
            with open(self.graph_output_path, 'a+') as f:
                f.write(content)
        # remove the intermediate results
        shutil.rmtree(self.tmp_graph_base_dir)


def main():
    # TODO: [Refactor] combine with pipeline abstraction KG builder
    # TODO: [Refactor] read column profiles path from project config.py
    # TODO: [Refactor] add graph output path to project config.py
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--column-profiles-path', type=str,  
                        default='../../profiler/src/storage/metadata/profiles', help='Path to column profiles')
    parser.add_argument('--out-graph-path', type=str, default='out/kglids_data_items_graph.nq',
                        help='Path to save the graph, including graph file name.')
    parser.add_argument('--spark-mode', type=str, default='local', help="Possible values: 'local' or 'cluster'")
    parser.add_argument('--memory-size', type=int, default=24, help='RAM in GB to be used for local mode.')
    args = parser.parse_args()

    start_all = datetime.now()
    knowledge_graph_builder = KnowledgeGraphBuilder(column_profiles_path=args.column_profiles_path,
                                                    out_graph_path=args.out_graph_path, 
                                                    spark_mode=args.spark_mode, memory_size=args.memory_size)

    # Membership (e.g. table -> dataset) and metadata (e.g. min, max) triples
    print(datetime.now(), "• 1. Building Membership and Metadata triples\n")
    start_schema = datetime.now()
    knowledge_graph_builder.build_membership_and_metadata_subgraph()
    end_schema = datetime.now()
    print(datetime.now(), "• Done.\tTime taken: " + str(end_schema - start_schema), '\n')

    print(datetime.now(), "• 2. Computing column-column similarities\n")
    start_schema_sim = datetime.now()
    knowledge_graph_builder.generate_similarity_triples()
    end_schema_sim = datetime.now()
    print(datetime.now(), "• Done.\tTime taken: " + str(end_schema_sim - start_schema_sim), '\n')

    print(datetime.now(), '• 3. Combining intermediate subgraphs from workers\n')
    knowledge_graph_builder.build_graph()

    print(datetime.now(), '\n• Done. Graph Saved to: out/kglids_data_items_graph.nt\n')

    end_all = datetime.now()
    print(datetime.now(), "Total time to build graph: " + str(end_all - start_all))


# imports
class Label:

    def __init__(self, text: str, lan: str):
        self.text = text
        self.lan = lan

    def get_text(self) -> str:
        return self.text

    def get_lan(self) -> str:
        return self.lan

    def __repr__(self):
        return "\"" + self.text + "\"" + "@" + self.lan

    def __str__(self):
        return str(self.__repr__())


class RDFResource:
    def __init__(self, content, namespace=None, isBlank=False):
        self.content = content
        self.isBlank = isBlank
        self.namespace = namespace

    def __repr__(self):
        if self.namespace:
            return "<{}{}>".format(self.namespace, self.content)

        if self.isBlank:
            return '_:{}'.format(self.content)
        if isinstance(self.content, str):
            return '\"{}\"'.format(self.content)
        if isinstance(self.content, int):
            return '\"{}\"^^xsd:integer'.format(self.content)
        if isinstance(self.content, float):
            return '\"{}\"^^xsd:double'.format(round(self.content, 3))
        if isinstance(self.content, Label):
            return str(self.content)

        return self.content

    def __str__(self):
        return str(self.__repr__())


class Triplet:
    # TODO: [Refactor] Rename to RDFTriple
    def __init__(self, rdf_subject, rdf_predicate, rdf_object):
        self.rdf_subject = rdf_subject
        self.rdf_predicate = rdf_predicate
        self.rdf_object = rdf_object

    def __repr__(self):
        return self.__repr_helper(True)

    def __repr_helper(self, isRoot):
        formattedSubject = self.rdf_subject
        formattedObject = self.rdf_object
        if isinstance(self.rdf_subject, Triplet):
            formattedSubject = '<<{}>>'.format(self.rdf_subject.__repr_helper(False))
        if isinstance(self.rdf_object, Triplet):
            formattedObject = '<<{}>>'.format(self.rdf_object.__repr_helper(False))

        if isRoot:
            return '{} {} {}.'.format(formattedSubject, self.rdf_predicate, formattedObject)
        else:
            return '{} {} {}'.format(formattedSubject, self.rdf_predicate, formattedObject)

    def __str__(self):
        return str(self.__repr__())

    def get_reversed_triple(self):
        # TODO: [Refactor] a better name for this method?
        # The reverse of an RDF-star triple, is the reverse of the subject
        if isinstance(self.rdf_subject, Triplet):
            return Triplet(self.rdf_subject.get_reversed_triple(), self.rdf_predicate, self.rdf_object)

        return Triplet(self.rdf_object, self.rdf_predicate, self.rdf_subject)


def column_metadata_worker(column_profile_paths, ontology, triples_output_tmp_dir):
    # TODO: [Refactor] have the names of predicates read from global project ontology object
    triples = []

    for column_profile_path in column_profile_paths:
        profile = ColumnProfile.load_profile(column_profile_path)

        column_node = RDFResource(profile.get_column_id(), ontology['kglidsResource'])
        table_node = RDFResource(profile.get_table_id(), ontology['kglidsResource'])
        col_label = generate_label(profile.get_column_name(), 'en')

        # membership
        triples.append(Triplet(column_node, RDFResource('isPartOf', ontology['kglids']), table_node))
        triples.append(Triplet(column_node, RDFResource('type', ontology['rdf']),
                               RDFResource('Column', ontology['kglids'])))
        # metadata
        triples.append(Triplet(column_node, RDFResource('hasDataType', ontology['kglidsData']),
                               RDFResource(profile.get_data_type())))
        triples.append(Triplet(column_node, RDFResource('name', ontology['schema']),
                               RDFResource(profile.get_column_name())))
        triples.append(Triplet(column_node, RDFResource('hasTotalValueCount', ontology['kglidsData']),
                               RDFResource(profile.get_total_values_count())))
        triples.append(Triplet(column_node, RDFResource('hasDistinctValueCount', ontology['kglidsData']),
                               RDFResource(profile.get_distinct_values_count())))
        triples.append(Triplet(column_node, RDFResource('hasMissingValueCount', ontology['kglidsData']),
                               RDFResource(profile.get_missing_values_count())))
        triples.append(Triplet(column_node, RDFResource('label', ontology['rdfs']), RDFResource(col_label)))

        if profile.is_numeric():
            triples.append(Triplet(column_node, RDFResource('hasMedianValue', ontology['kglidsData']),
                                   RDFResource(profile.get_median())))
            triples.append(Triplet(column_node, RDFResource('hasMaxValue', ontology['kglidsData']),
                                   RDFResource(profile.get_max_value())))
            triples.append(Triplet(column_node, RDFResource('hasMinValue', ontology['kglidsData']),
                                   RDFResource(profile.get_min_value())))

    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
    with open(os.path.join(triples_output_tmp_dir, filename), 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(f"{triple}\n")

    return []


def column_pair_similarity_worker(column_idx, column_profile_paths, ontology, triples_output_tmp_dir,
                                  semantic_similarity_threshold, numerical_content_threshold,
                                  deep_embedding_content_threshold, inclusion_dependency_threshold,
                                  minhash_content_threshold, pkfk_threshold, word_embedding):
    # load the column profiles
    column1_profile = ColumnProfile.load_profile(column_profile_paths[column_idx])
    similarity_triples = []
    for j in range(column_idx + 1, len(column_profile_paths)):
        column2_profile = ColumnProfile.load_profile(column_profile_paths[j])

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

        similarity_triples.extend(
            semantic_triples + content_triples + deep_content_triples + inclusion_triples + pkfk_triples)
    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
    with open(os.path.join(triples_output_tmp_dir, filename), 'w', encoding='utf-8') as f:
        for triple in similarity_triples:
            f.write(f"{triple}\n")

    return []


def _compute_semantic_similarity(column1_profile: ColumnProfile, column2_profile: ColumnProfile, ontology,
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


def _compute_content_similarity(col1_profile: ColumnProfile, col2_profile: ColumnProfile, ontology,
                                numerical_content_threshold, minhash_content_threshold):
    # TODO: [Refactor] have the names of predicates read from global project ontology object
    content_similarity_triples = []
    if col1_profile.is_numeric():

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


def _compute_numerical_deep_content_similarity(col1_profile: ColumnProfile, col2_profile: ColumnProfile, ontology,
                                               deep_embedding_content_threshold):
    deep_content_similarity_triples = []
    if col1_profile.is_numeric():
        # Numerical column: calculate similarity with and without deep embeddings
        embedding_cos_sim = np.dot(col1_profile.get_deep_embedding(), col2_profile.get_deep_embedding()) / \
                            (norm(col1_profile.get_deep_embedding()) * norm(col2_profile.get_deep_embedding()))
        if embedding_cos_sim >= deep_embedding_content_threshold:
            deep_content_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                                     'hasDeepEmbeddingContentSimilarity',
                                                                                     embedding_cos_sim,
                                                                                     ontology))
    return deep_content_similarity_triples


def _compute_numerical_inclusion_dependency(col1_profile: ColumnProfile, col2_profile: ColumnProfile, ontology,
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
def _compute_inclusion_overlap(column1_profile: ColumnProfile, column2_profile: ColumnProfile):
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


def _compute_primary_key_foreign_key_similarity(col1_profile: ColumnProfile, col2_profile: ColumnProfile, ontology,
                                                pkfk_threshold, content_similarity_triples,
                                                deep_content_similarity_triples, inclusion_dependency_triples):
    # we have pkfk if the two columns have content similarity and their cardinalities are above the provided threshold
    if col1_profile.get_total_values_count() == 0 or col2_profile.get_total_values_count() == 0:
        return []
    col1_cardinality = float(col1_profile.get_distinct_values_count()) / float(col1_profile.get_total_values_count())
    col2_cardinality = float(col2_profile.get_distinct_values_count()) / float(col2_profile.get_total_values_count())
    if min(col1_cardinality, col2_cardinality) < pkfk_threshold:
        return []

    highest_cardinality = max(col1_cardinality, col2_cardinality)
    pkfk_similarity_triples = []
    if inclusion_dependency_triples or (col1_profile.is_textual() and content_similarity_triples):
        pkfk_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                         'hasPrimaryKeyForeignKeySimilarity',
                                                                         highest_cardinality, ontology))
    if deep_content_similarity_triples:
        pkfk_similarity_triples.extend(_create_column_similarity_triples(col1_profile, col2_profile,
                                                                         'hasDeepPrimaryKeyForeignKeySimilarity',
                                                                         highest_cardinality, ontology))

    return pkfk_similarity_triples


# TODO: [Refactor] this method needs to be moved somewhere else.
# TODO: [Refactor] this method shouldn't take ontology from the worker methods. It should be passed from global config
def _create_column_similarity_triples(column1_profile: ColumnProfile, column2_profile: ColumnProfile,
                                      similarity_predicate, similarity_score, ontology):
    nested_subject = RDFResource(column1_profile.get_column_id(), ontology['kglidsResource'])
    nested_predicate = RDFResource(similarity_predicate, ontology['kglidsData'])
    nested_object = RDFResource(column2_profile.get_column_id(), ontology['kglidsResource'])
    # RDF-star triple of the content similarity between the two columns with the score.
    triple = Triplet(Triplet(nested_subject, nested_predicate, nested_object),
                     RDFResource('withCertainty', ontology['kglidsData']), RDFResource(similarity_score))
    return [triple, triple.get_reversed_triple()]


class WordEmbeddings:
    def __init__(self, word_embeddings_path):
        self.raw_word_embeddings_path = word_embeddings_path
        self.normalized_word_embeddings_path = self.raw_word_embeddings_path[
                                               :self.raw_word_embeddings_path.rindex('.')] + '.pickle'
        # initialize and normalize the word vectors if they don't exist
        if not os.path.exists(self.normalized_word_embeddings_path):
            self._initialize_and_normalize_word_embeddings()

        with open(self.normalized_word_embeddings_path, 'rb') as f:
            print('Loading:', self.normalized_word_embeddings_path)
            self.vectors = pickle.load(f)

    def _initialize_and_normalize_word_embeddings(self):
        # loads raw word embedding file, normalizes the vectors to unit length, and stores them as pickle
        print('initializing raw word embeddings and normalizing them to unit length')
        with open(self.raw_word_embeddings_path, 'r') as f:
            lines = [i.strip() for i in f.readlines()]
        vectors = {}
        for line in lines:
            split = line.split()
            word = split[0]
            vector = np.array([float(i) for i in split[1:]])
            vector /= np.linalg.norm(vector)  # normalize to unit length
            vectors[word] = vector
        with open(self.normalized_word_embeddings_path, 'wb') as f:
            pickle.dump(vectors, f)

    def semantic_distance(self, v1, v2):
        if v1 is None or v2 is None:
            print("unknowns")
            return -99
        else:
            v1 = np.array(v1)
            v2 = np.array(v2)
            sim = np.dot(v1, v2.T)
        return sim

    def get_distance_between_column_labels(self, l1, l2):
        l1_tokens = l1.split(' ')
        l2_tokens = l2.split(' ')
        if l1 == l2:
            return 1.0
        if l1 == '' and l2 != '' or l1 != '' and l2 == '':
            return 0.0
        intersection = set(l1_tokens).intersection(l2_tokens)
        # remove common token
        if len(l1_tokens) > 1 and len(l2_tokens) > 1:
            l1_tokens = [i for i in l1_tokens if i not in intersection]
            l2_tokens = [i for i in l2_tokens if i not in intersection]

        if len(l1_tokens) >= 1 and len(l2_tokens) == 0 or len(l2_tokens) >= 1 and len(l1_tokens) == 0:
            l1_tokens = l1.split(' ')
            l2_tokens = l2.split(' ')
        word_embeddings_for_tokens = self.get_embeddings_for_tokens(l1_tokens + l2_tokens)

        combinations = list(itertools.product(l1_tokens, l2_tokens))
        distance = 0.0
        for v1, v2 in combinations:
            emb1 = word_embeddings_for_tokens[v1]
            emb2 = word_embeddings_for_tokens[v2]
            if emb1 is None or emb2 is None:
                return 0

            d = self.semantic_distance(emb1, emb2)
            distance += d
        return (distance / len(combinations)) if combinations else 1

    def get_embeddings_for_tokens(self, tokens):
        # TODO: [Implement] does it make sense to take only the first taken?
        embeddings = {i: self.vectors.get(i.strip().split()[0], None) for i in tokens}
        return embeddings

def generate_label(col_name: str, lan: str) -> Label:
    # TODO: [Implement] the way labels are generated is not 100% the best. It is not always best to split by camel case
    if '.csv' in col_name:
        col_name = re.sub('.csv', '', col_name)
    col_name = re.sub('[^0-9a-zA-Z]+', ' ', col_name)
    text = " ".join(camelsplit(col_name.strip()))
    text = re.sub('\s+', ' ', text.strip())
    return Label(text.lower(), lan)


class ColumnProfile:
    # TODO: [Refactor] remove unneeded stats
    # TODO: [Refactor] combine minhash and deep_embedding attribute to embedding (minhash for strings and DDE for numerical)    
    def __init__(self, column_id: float, origin: str, dataset_name: str, dataset_id: str, path: str, table_name: str,
                 table_id: str, column_name: str, datasource: str, data_type: str,
                 total_values: float, distinct_values_count: float, missing_values_count: float, min_value: float,
                 max_value: float, mean: float, median: float, iqr: float,
                 minhash: list, deep_embedding: list):
        self.column_id = column_id
        self.origin = origin
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.path = path
        self.table_name = table_name
        self.table_id = table_id
        self.datasource = datasource
        self.column_name = column_name
        self.data_type = data_type
        self.total_values = total_values
        self.distinct_values_count = distinct_values_count
        self.missing_values_count = missing_values_count
        self.max_value = max_value
        self.min_value = min_value
        self.mean = mean
        self.median = median
        self.iqr = iqr
        self.minhash = minhash
        self.deep_embedding = deep_embedding

    def to_dict(self):
        # TODO: [Refactor] rename these keys
        profile_dict = {'column_id': self.get_column_id(),
                        'origin': self.get_origin(),
                        'datasetName': self.get_dataset_name(),
                        'datasetid': self.get_dataset_id(),
                        'path': self.get_path(),
                        'tableName': self.get_table_name(),
                        'tableid': self.get_table_id(),
                        'columnName': self.get_column_name(),
                        'datasource': self.get_datasource(),
                        'dataType': self.get_data_type(),
                        'totalValuesCount': self.get_total_values_count(),
                        'distinctValuesCount': self.get_distinct_values_count(),
                        'missingValuesCount': self.get_missing_values_count(),
                        'minValue': self.get_min_value(),
                        'maxValue': self.get_max_value(),
                        'avgValue': self.get_mean(),
                        'median': self.get_median(),
                        'iqr': self.get_iqr(),
                        'minhash': self.get_minhash(),
                        'deep_embedding': self.get_deep_embedding()}
        return profile_dict

    def save_profile(self, column_profile_base_dir):
        profile_save_path = os.path.join(column_profile_base_dir, self.get_data_type())
        os.makedirs(profile_save_path, exist_ok=True)
        # random generated name of length 10 to avoid synchronization between threads and profile name collision
        profile_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        with open(os.path.join(profile_save_path, f'{profile_name}.json'), 'w') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_profile(column_profile_path):
        with open(column_profile_path) as f:
            profile_dict = json.load(f)
        profile = ColumnProfile(column_id=profile_dict.get('column_id'),
                                origin=profile_dict.get('origin'),
                                dataset_name=profile_dict.get('datasetName'),
                                dataset_id=profile_dict.get('datasetid'),
                                path=profile_dict.get('path'),
                                table_name=profile_dict.get('tableName'),
                                table_id=profile_dict.get('tableid'),
                                column_name=profile_dict.get('columnName'),
                                datasource=profile_dict.get('datasource'),
                                data_type=profile_dict.get('dataType'),
                                total_values=profile_dict.get('totalValuesCount'),
                                distinct_values_count=profile_dict.get('distinctValuesCount'),
                                missing_values_count=profile_dict.get('missingValuesCount'),
                                min_value=profile_dict.get('minValue'),
                                max_value=profile_dict.get('maxValue'),
                                mean=profile_dict.get('avgValue'),
                                median=profile_dict.get('median'),
                                iqr=profile_dict.get('iqr'),
                                minhash=profile_dict.get('minhash'),
                                deep_embedding=profile_dict.get('deep_embedding'))
        return profile

    # TODO: [Refactor] have the data types in a global project config
    def is_numeric(self) -> bool:
        return self.get_data_type() in ['N']

    def is_textual(self) -> bool:
        return self.get_data_type() in ['T', 'T_code', 'T_date', 'T_email', 'T_loc', 'T_org', 'T_person']

    def is_boolean(self) -> bool:
        return self.get_data_type() in ['B']

    def get_column_id(self) -> float:
        return self.column_id

    def get_origin(self) -> str:
        return self.origin

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def get_dataset_id(self) -> str:
        return self.dataset_id

    def get_path(self) -> str:
        return self.path

    def get_table_name(self) -> str:
        return self.table_name

    def get_table_id(self) -> str:
        return self.table_id

    def get_column_name(self) -> str:
        return self.column_name

    def get_datasource(self) -> str:
        return self.datasource

    def get_data_type(self) -> str:
        return self.data_type

    def get_total_values_count(self) -> float:
        return self.total_values

    def get_distinct_values_count(self) -> float:
        return self.distinct_values_count

    def get_missing_values_count(self) -> float:
        return self.missing_values_count

    def get_minhash(self) -> list:
        return self.minhash

    def get_deep_embedding(self) -> list:
        return self.deep_embedding

    def get_min_value(self) -> float:
        return self.min_value

    def get_max_value(self) -> float:
        return self.max_value

    def get_mean(self) -> float:
        return self.mean

    def get_median(self) -> float:
        return self.median

    def get_iqr(self) -> float:
        return self.iqr

    def set_column_id(self, column_id: float):
        self.column_id = column_id

    def set_origin(self, origin: str):
        self.origin = origin

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_path(self, path: str):
        self.path = path

    def set_table_name(self, table_name: str):
        self.table_name = table_name

    def set_column_name(self, column_name: str):
        self.column_name = column_name

    def set_data_type(self, data_type: str):
        self.data_type = data_type

    def set_total_values(self, total_values: float):
        self.total_values = total_values

    def set_distinct_values_count(self, unique_values: float):
        self.distinct_values_count = unique_values

    def set_missing_values_count(self, missing_values_count: float):
        self.missing_values_count = missing_values_count

    def set_min_value(self, min_value: float):
        self.min_value = min_value

    def set_max_value(self, max_value: float):
        self.max_value = max_value

    def set_mean(self, mean: float):
        self.mean = mean

    def set_median(self, median: float):
        self.median = median

    def set_iqr(self, iqr: float):
        self.iqr = iqr

    def set_minhash(self, minhash: list):
        self.minhash = minhash

    def set_deep_embedding(self, deep_embedding: list):
        self.deep_embedding = deep_embedding

    def __str__(self):
        return self.table_name + ': ' + str(self.minhash) if self.minhash else str(self.deep_embedding)

if __name__ == '__main__':
    main()
