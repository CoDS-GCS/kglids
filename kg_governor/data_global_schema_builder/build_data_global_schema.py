import argparse
from datetime import datetime
import glob
import multiprocessing as mp
import os
import random
import string
import shutil

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from tqdm import tqdm

from kg_governor.data_global_schema_builder.workers import column_metadata_worker, column_pair_similarity_worker
from kg_governor.data_global_schema_builder.utils.word_embeddings import WordEmbeddings
from kg_governor.data_global_schema_builder.utils.utils import generate_label, RDFResource, Triplet
from kg_governor.data_profiling.model.column_profile import ColumnProfile
from kglids_config import KGLiDSConfig


class DataGlobalSchemaBuilder:
    def __init__(self, column_profiles_path, out_graph_path, is_spark_local_mode, label_sim_threshold,
                 embedding_sim_threshold, boolean_sim_threshold):
        self.column_profiles_base_dir = column_profiles_path
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

        self.memory_size = (os.sysconf('SC_PAGE_SIZE') * os.sysconf(
            'SC_PHYS_PAGES') // 1024 ** 3) - 1  # total RAM - 1 GB
        self.is_spark_local_mode = is_spark_local_mode

        self.label_sim_threshold = label_sim_threshold
        self.embedding_sim_threshold = embedding_sim_threshold
        self.boolean_sim_threshold = boolean_sim_threshold

        self.ontology = {'kglids': 'http://kglids.org/ontology/',
                         'kglidsData': 'http://kglids.org/ontology/data/',
                         'kglidsResource': 'http://kglids.org/resource/',
                         'schema': 'http://schema.org/',
                         'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                         'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'}

        # get list of column profiles and their data type
        column_data_types = [str(i.name) for i in os.scandir(column_profiles_path) if i.is_dir()]
        column_profile_paths = []
        print('Column Type Breakdown:')
        for data_type in column_data_types:
            type_path = os.path.join(column_profiles_path, data_type)
            profiles = [os.path.join(type_path, i) for i in os.listdir(type_path) if i.endswith('.json')]
            column_profile_paths.extend(profiles)
            print(f'\t{data_type}: {len(profiles)}')
        print('Total:', len(column_profile_paths))
        print('Reading column profiles ...')
        pool = mp.Pool(os.cpu_count() - 1)
        self.column_profiles = list(tqdm(pool.imap_unordered(ColumnProfile.load_profile, column_profile_paths),
                                         total=len(column_profile_paths)))

        self.word_embedding_path = os.path.join(KGLiDSConfig.base_dir, 'storage/embeddings/glove.6B.100d.txt')
        # make sure the word embeddings are initialized
        self.word_embedding = WordEmbeddings(self.word_embedding_path)

        if not self.is_spark_local_mode:
            self.spark = (SparkSession.builder
                          .appName("KGBuilder")
                          .getOrCreate()
                          .sparkContext)
            # TODO: [Refactor] make sure this is updated. 
            # add python dependencies
            for pyfile in glob.glob('./**/*.py', recursive=True):
                self.spark.addPyFile(pyfile)
            # self.spark.addPyFile('../../data_profiling/src/data/column_profile.py')
        else:
            self.spark = SparkContext(conf=SparkConf().setMaster(f'local[*]')
                                      .set('spark.driver.memory', f'{self.memory_size}g'))

    def build_membership_and_metadata_subgraph(self):

        # generate column metadata triples in parallel
        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        # mapPartitions so we don't end up with too many subgraph files (compared to .map())
        column_profile_paths_rdd = self.spark.parallelize(self.column_profiles)
        column_profile_paths_rdd.mapPartitions(lambda x: column_metadata_worker(column_profiles=x,
                                                                                ontology=ontology,
                                                                                triples_output_tmp_dir=tmp_graph_dir)) \
            .collect()
        # generate table and dataset membership triples
        membership_triples = []
        tables = set()
        datasets = set()
        sources = set()
        for column_profile in self.column_profiles:
            if column_profile.get_table_id() in tables:
                continue
            # table -> dataset membership and metadata
            tables.add(column_profile.get_table_id())
            table_node = RDFResource(column_profile.get_table_id(), self.ontology['kglidsResource'])
            dataset_node = RDFResource(column_profile.get_dataset_id(), self.ontology['kglidsResource'])
            table_label = generate_label(column_profile.get_table_name(), 'en')
            membership_triples.append(Triplet(table_node, RDFResource('isPartOf', self.ontology['kglids']),
                                              RDFResource(dataset_node)))
            membership_triples.append(Triplet(table_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(column_profile.get_table_name())))
            membership_triples.append(Triplet(table_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(table_label)))
            membership_triples.append(Triplet(table_node, RDFResource('hasFilePath', self.ontology['kglidsData']),
                                              RDFResource(column_profile.get_path())))
            membership_triples.append(Triplet(table_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Table', self.ontology['kglids'])))

            if column_profile.get_dataset_id() in datasets:
                continue
            # dataset -> source membership and metadata
            datasets.add(column_profile.get_dataset_id())
            source_node = RDFResource(column_profile.get_data_source(), self.ontology['kglidsResource'])
            membership_triples.append(Triplet(dataset_node, RDFResource('isPartOf', self.ontology['kglids']),
                                              RDFResource(source_node)))
            dataset_label = generate_label(column_profile.get_dataset_name(), 'en')
            membership_triples.append(Triplet(dataset_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(column_profile.get_dataset_name())))
            membership_triples.append(Triplet(dataset_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(dataset_label)))
            membership_triples.append(Triplet(dataset_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Dataset', self.ontology['kglids'], False)))

            if column_profile.get_data_source() in sources:
                continue
            # source metadata
            sources.add(column_profile.get_data_source())
            source_label = generate_label(column_profile.get_data_source(), 'en')
            membership_triples.append(Triplet(source_node, RDFResource('name', self.ontology['schema']),
                                              RDFResource(column_profile.get_data_source())))
            membership_triples.append(Triplet(source_node, RDFResource('label', self.ontology['rdfs']),
                                              RDFResource(source_label)))
            membership_triples.append(Triplet(source_node, RDFResource('type', self.ontology['rdf']),
                                              RDFResource('Source', self.ontology['kglids'], False)))
        filename = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + '.nt'
        with open(os.path.join(self.tmp_graph_base_dir, filename), 'w', encoding='utf-8') as f:
            for triple in membership_triples:
                f.write(f"{triple}\n")

    def generate_similarity_triples(self):

        column_profiles = self.column_profiles
        ontology = self.ontology
        tmp_graph_dir = self.tmp_graph_base_dir
        word_embedding = self.word_embedding
        column_profile_indexes = list(range(len(self.column_profiles)))
        random.shuffle(column_profile_indexes)
        column_profile_indexes_rdd = self.spark.parallelize(column_profile_indexes)
        label_sim_threshold = self.label_sim_threshold
        embedding_sim_threshold = self.embedding_sim_threshold
        boolean_sim_threshold = self.boolean_sim_threshold
        column_profile_indexes_rdd.map(
            lambda x: column_pair_similarity_worker(column_idx=x,
                                                    column_profiles=column_profiles,
                                                    ontology=ontology,
                                                    triples_output_tmp_dir=tmp_graph_dir,
                                                    label_sim_threshold=label_sim_threshold,
                                                    embedding_sim_threshold=embedding_sim_threshold,
                                                    boolean_sim_threshold=boolean_sim_threshold,
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


def build_data_global_schema():
    start_all = datetime.now()
    knowledge_graph_builder = DataGlobalSchemaBuilder(column_profiles_path=KGLiDSConfig.profiles_out_path,
                                                      out_graph_path=KGLiDSConfig.data_global_schema_graph_out_path,
                                                      is_spark_local_mode=KGLiDSConfig.is_spark_local_mode,
                                                      label_sim_threshold=KGLiDSConfig.col_label_sim_threshold,
                                                      embedding_sim_threshold=KGLiDSConfig.col_embedding_sim_threshold,
                                                      boolean_sim_threshold=KGLiDSConfig.col_boolean_sim_threshold)

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

    end_all = datetime.now()
    print(datetime.now(), f'\n• Done. Graph Saved to: {KGLiDSConfig.data_global_schema_graph_out_path}.')

    print(datetime.now(), "Done. Total time to build graph: " + str(end_all - start_all))
