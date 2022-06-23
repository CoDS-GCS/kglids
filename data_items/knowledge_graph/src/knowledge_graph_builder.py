import os
import shutil
from datetime import datetime
import itertools
import random
import string
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../../')

# from pyspark import SparkConf, SparkContext

from rdf_resource import RDFResource
from triplet import Triplet
from workers import column_metadata_worker, column_pair_similarity_worker
from word_embedding.word_embeddings import WordEmbeddings
from utils import generate_label
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


def _metadata_worker_helper(args):
    column_profile_paths, ontology, tmp_graph_dir = args
    column_metadata_worker(column_profile_paths=column_profile_paths, ontology=ontology,
                           triples_output_tmp_dir=tmp_graph_dir)

def _similarity_worker_helper(args):
    column_profile_path_pairs, ontology, tmp_graph_dir, word_embedding_path = args
    column_pair_similarity_worker(column_profile_path_pairs=column_profile_path_pairs,
                                  ontology=ontology,
                                  triples_output_tmp_dir=tmp_graph_dir,
                                  semantic_similarity_threshold=SEMANTIC_THRESHOLD,
                                  numerical_content_threshold=CONTENT_THRESHOLD,
                                  deep_embedding_content_threshold=DEEP_EMBEDDING_THRESHOLD,
                                  inclusion_dependency_threshold=INCLUSION_THRESHOLD,
                                  minhash_content_threshold=MINHASH_THRESHOLD,
                                  pkfk_threshold=PKFK_THRESHOLD,
                                  word_embedding_path=word_embedding_path)


class KnowledgeGraphBuilder:
    # TODO: [Refactor] add all kglids URIs to knowledge graph config.py
    # TODO: [Refactor] have Spark configuration read from the global project config
    # TODO: [Refactor] read raw word embeddings path from global project config
    def __init__(self, column_profiles_path, out_graph_path):
        self.graph_output_path = out_graph_path
        self.out_graph_base_dir = os.path.dirname(self.graph_output_path)
        self.tmp_graph_base_dir = os.path.join(self.out_graph_base_dir, 'tmp')  # for intermediate results
        if os.path.exists(self.tmp_graph_base_dir):
            shutil.rmtree(self.tmp_graph_base_dir)
        os.makedirs(self.tmp_graph_base_dir)

        # memory_gb = 24
        # conf = (SparkConf()
        #         .setMaster(f'local[*]')
        #         .set('spark.driver.memory', f'{memory_gb}g'))
        # self.spark = SparkContext(conf=conf)

        self.ontology = {'kglids': 'http://kglids.org/ontology/',
                         'kglidsData': 'http://kglids.org/ontology/data/',
                         'kglidsResource': 'http://kglids.org/resource/',
                         'schema': 'http://schema.org/',
                         'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                         'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'}

        # get list of column profiles and their data type
        column_data_types = [i for i in os.listdir(column_profiles_path)]
        self.column_profile_paths_per_type = {}
        for data_type in column_data_types:
            type_path = os.path.join(column_profiles_path, data_type)
            profiles = [os.path.join(type_path, i) for i in os.listdir(type_path) if i.endswith('.json')]
            self.column_profile_paths_per_type[data_type] = profiles

        print('Column Type Breakdown:')
        for data_type, profile_paths in self.column_profile_paths_per_type.items():
            print(f'\t{data_type}: {len(profile_paths)}')

        # TODO: [Refactor] Read this from project config
        self.word_embedding_path = '../../data/glove.6B.100d.txt'
        # make sure the word embeddings are initialized
        word_embedding = WordEmbeddings(self.word_embedding_path)
        word_embedding = None

    def build_membership_and_metadata_subgraph(self):
        column_profile_paths = list(itertools.chain.from_iterable(self.column_profile_paths_per_type.values()))
        # columns_rdd = self.spark.parallelize(column_profile_paths)
        # generate column metadata triples in parallel
        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        # mapPartitions so we don't end up with too many subgraph files (compared to .map())
        # columns_rdd.mapPartitions(lambda x: column_metadata_worker(column_profile_paths=x, ontology=ontology,
        #                                                            triples_output_tmp_dir=tmp_graph_dir)).collect()
        # TODO: TMP Change
        column_profile_paths_grouped = [(i, ontology, tmp_graph_dir) for i in np.array_split(column_profile_paths, os.cpu_count())]
        pool = mp.Pool(os.cpu_count())
        list(tqdm(pool.imap_unordered(_metadata_worker_helper, column_profile_paths_grouped), total=len(column_profile_paths_grouped)))
        
        # generate table and dataset membership triples
        membership_triples = []
        tables = set()
        datasets = set()
        sources = set()
        for column_profile_path in column_profile_paths:
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
        # create column profile pairs per type
        column_pairs = []
        for data_type, column_profiles in self.column_profile_paths_per_type.items():
            for i in range(len(column_profiles)):
                for j in range(i + 1, len(column_profiles)):
                    column_pairs.append((column_profiles[i], column_profiles[j]))

        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        word_embedding_path = self.word_embedding_path
        # column_pairs_rdd = self.spark.parallelize(column_pairs)
        # mapPartitions so we don't end up with too many subgraph files
        # column_pairs_rdd.mapPartitions(lambda x: column_pair_similarity_worker(column_profile_path_pairs=x,
        #                                                                        ontology=ontology,
        #                                                                        triples_output_tmp_dir=tmp_graph_dir,
        #                                                                        semantic_similarity_threshold=SEMANTIC_THRESHOLD,
        #                                                                        numerical_content_threshold=CONTENT_THRESHOLD,
        #                                                                        deep_embedding_content_threshold=DEEP_EMBEDDING_THRESHOLD,
        #                                                                        inclusion_dependency_threshold=INCLUSION_THRESHOLD,
        #                                                                        minhash_content_threshold=MINHASH_THRESHOLD,
        #                                                                        pkfk_threshold=PKFK_THRESHOLD,
        #                                                                        word_embedding_path=word_embedding_path)) \
        #     .collect()
        column_pairs = [(i, ontology, tmp_graph_dir, word_embedding_path) for i in np.array_split(column_pairs, os.cpu_count())]
        pool = mp.Pool(os.cpu_count())
        list(tqdm(pool.imap_unordered(_similarity_worker_helper, column_pairs), total=len(column_pairs)))

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

    if os.path.exists('out/kglids_data_items_graph.nt'):
        os.remove('out/kglids_data_items_graph.nt')

    start_all = datetime.now()
    knowledge_graph_builder = KnowledgeGraphBuilder(
        column_profiles_path='../../profiler/src/storage/metadata/profiles',
        out_graph_path='out/kglids_data_items_graph.nq')

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


if __name__ == '__main__':
    main()
