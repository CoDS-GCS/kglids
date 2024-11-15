import argparse
from datetime import datetime
import glob
import multiprocessing as mp
import os
import random
import shutil
import string
import sys
import requests
import json
import shutil

sys.path.append('../../../')

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from tqdm import tqdm

# TODO: [Refactor] project structure needs to be changed. These imports won't work in terminal without the sys call.
from workers import column_metadata_worker, column_pair_similarity_worker
from kg_governor.knowledge_graph_construction.src.utils.word_embeddings import WordEmbeddings
from kg_governor.knowledge_graph_construction.src.utils.utils import generate_label, RDFResource, Triplet
from kg_governor.data_profiling.src.model.column_profile import ColumnProfile


class DataGlobalSchemaBuilder:
    # TODO: [Refactor] add all kglids URIs to knowledge graph config.py
    # TODO: [Refactor] have Spark configuration read from the global project config
    # TODO: [Refactor] read raw word embeddings path from global project config
    def __init__(self, column_profiles_path, out_graph_path, spark_mode, label_sim_threshold,
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

        self.memory_size = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 1024**3) - 1 # total RAM - 1 GB
        self.is_cluster_mode = spark_mode == 'cluster'

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
        

        # TODO: [Refactor] Read this from project config
        self.word_embedding_path = 'utils/glove_embeddings/glove.6B.100d.txt'
        # make sure the word embeddings are initialized
        self.word_embedding = WordEmbeddings(self.word_embedding_path)
        
        if self.is_cluster_mode:
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
        column_profiles_base_dir = self.column_profiles_base_dir
        is_cluster_mode = self.is_cluster_mode
        # mapPartitions so we don't end up with too many subgraph files (compared to .map())
        column_profile_paths_rdd = self.spark.parallelize(self.column_profiles)
        column_profile_paths_rdd.mapPartitions(lambda x: column_metadata_worker(column_profiles=x,
                                                                                ontology=ontology,
                                                                                triples_output_tmp_dir=tmp_graph_dir))\
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


def main():
    # TODO: [Refactor] combine with pipeline abstraction KG builder
    # TODO: [Refactor] read column profiles path from project config.py
    # TODO: [Refactor] add graph output path to project config.py

    # ************* SYSTEM PARAMETERS**********************
    # TODO: [Refactor] have these inside a global project config
    DEFAULT_LABEL_SIM_THRESHOLD = 0.75
    DEFAULT_BOOLEAN_SIM_THRESHOLD = 0.75
    DEFAULT_EMBEDDING_SIM_THRESHOLD = 0.75
    # *****************************************************
    parser = argparse.ArgumentParser()
    parser.add_argument('--column-profiles-path', type=str,
                        default='../../../storage/profiles/smaller_real_profiles', help='Path to column profiles')
    parser.add_argument('--out-graph-path', type=str,
                        default='../../../storage/knowledge_graph/data_global_schema/data_global_schema_graph.ttl',
                        help='Path to save the graph, including graph file name.')
    parser.add_argument('--spark-mode', type=str, default='local', help="Possible values: 'local' or 'cluster'")
    parser.add_argument('--label-sim-threshold', type=float, default=DEFAULT_LABEL_SIM_THRESHOLD)
    parser.add_argument('--embedding-sim-threshold', type=float, default=DEFAULT_EMBEDDING_SIM_THRESHOLD)
    parser.add_argument('--boolean-sim-threshold', type=float, default=DEFAULT_BOOLEAN_SIM_THRESHOLD)
    parser.add_argument('--graphdb-endpoint', type=str, default='http://localhost:7200')
    parser.add_argument('--graphdb-import-dir', type=str, default=os.path.expanduser('~/graphdb-import/'))
    parser.add_argument('--graphdb-repo', type=str)
    parser.add_argument('--replace-existing-repo', type=bool, default=False)
    args = parser.parse_args()

    start_all = datetime.now()
    knowledge_graph_builder = DataGlobalSchemaBuilder(column_profiles_path=args.column_profiles_path,
                                                      out_graph_path=args.out_graph_path,
                                                      spark_mode=args.spark_mode,
                                                      label_sim_threshold=args.label_sim_threshold,
                                                      embedding_sim_threshold=args.embedding_sim_threshold,
                                                      boolean_sim_threshold=args.boolean_sim_threshold)

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

    print(datetime.now(), f'\n• Graph Saved to: {args.out_graph_path}\n')

    print(datetime.now(), '• 4. Loading graph to GraphDB at:', args.graphdb_endpoint, args.graphdb_repo, '\n')
    # check if repo with the same name exists
    url = args.graphdb_endpoint + '/rest/repositories'
    graphdb_repos = json.loads(requests.get(url).text)
    graphdb_repo_ids = [i['id'] for i in graphdb_repos]
    # remove existing repo if found
    headers = {"Content-Type": "application/json"}
    if args.graphdb_repo in graphdb_repo_ids:
        if args.replace_existing_repo:
            url = f"{args.graphdb_endpoint}/rest/repositories/{args.graphdb_repo}"
            response = requests.delete(url)
            if response.status_code // 100 != 2:
                print(datetime.now(), ': Error while deleting GraphDB repo:', args.graphdb_repo, ':', response.text)
    else:
        # create a new repo
        url = args.graphdb_endpoint + '/rest/repositories'
        data = {
            "id": args.graphdb_repo,
            "type": "graphdb",
            "title": args.graphdb_repo,
            "params": {
                "defaultNS": {
                    "name": "defaultNS",
                    "label": "Default namespaces for imports(';' delimited)",
                    "value": ""
                },
                "imports": {
                    "name": "imports",
                    "label": "Imported RDF files(';' delimited)",
                    "value": ""
                },
                "enableContextIndex": {
                    "name": "enableContextIndex",
                    "label": "Enable context index",
                    "value": "true"
                }
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code // 100 != 2:
            print(datetime.now(), "Error creating the GraphDB repo:", args.graphdb_repo, ':', response.text)
    # copy generated file to graphdb-import
    tmp_file_name = args.graphdb_repo + '_import.ttl'
    shutil.copy2(args.out_graph_path, os.path.join(args.graphdb_import_dir, tmp_file_name))
    # import copied graph
    url = f"{args.graphdb_endpoint}/rest/repositories/{args.graphdb_repo}/import/server"
    data = {"fileNames": [tmp_file_name]}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code // 100 != 2:
        print(datetime.now(), 'Error importing file:', tmp_file_name, 'to GraphDB repo:', args.graphdb_repo, ':',
              response.text)
    # remove copied graph
    # os.remove(os.path.join(args.graphdb_import_dir, tmp_file_name))

    end_all = datetime.now()
    print(datetime.now(), "Done. Graph is being uploaded to GraphDB. Total time to build graph: " + str(end_all - start_all))


if __name__ == '__main__':
    main()
