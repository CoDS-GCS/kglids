# Contains all project configs
import os
from pathlib import Path


class KGLiDSConfig:
    ####### Data source configs #######
    # name of the data source e.g. kaggle
    data_source = 'kaggle_small'
    # path containing datasets, each containing 'data' and 'notebooks' directories.
    data_source_path = '/home/mossad/projects/kglids/storage/data_sources/kaggle_small'

    ####### General configs #######
    # the base directory of the KGLiDS project, i.e. this directory
    base_dir = str(Path.resolve(Path(__file__).parent))
    # name of the corresponding repository on GraphDB to load the knowledge graph
    graphdb_repo_name = data_source
    # graphdb endpoint (defaults to http://localhost:7200 )
    graphdb_endpoint = 'http://localhost:7200'
    # graphdb server imports directory to load big graphs (defaults to ~/graphdb-import)
    graphdb_import_path = os.path.expanduser('~/graphdb-import/')
    # whether to replace existing GraphDB repository
    replace_existing_graphdb_repo = True
    # name of the database on Postgres to load the column embeddings
    pgvector_db_name = f'{data_source}_embeddings'

    ####### Spark configs #######
    # whether to run Spark in local or cluster mode.
    is_spark_local_mode = True
    # number of workers (processes) to use when profiling columns. Defaults to the number of threads.
    spark_n_workers = os.cpu_count()
    # maximum memory in GB to be used by Spark
    spark_max_memory = 25

    ####### Data profiling configs #######
    # directory to save the generated column profiles.
    profiles_out_path = os.path.join(base_dir, 'storage', 'profiles', f'{data_source}_profiles')
    # whether to replace existing profiles if found
    replace_existing_profiles = False

    ####### Pipeline abstraction configs #######
    # path to generate the pipeline subgraphs before loading it to GraphDB
    pipeline_graphs_out_path = os.path.join(base_dir, 'storage', 'pipeline_graphs', f'{data_source}_pipeline_graphs')

    ####### data global schema construction configs #######
    # path to generate the data global schema graph before loading it to GraphDB
    data_global_schema_graph_out_path = os.path.join(base_dir, 'storage', 'knowledge_graph', 'data_global_schema',
                                                     f'{data_source}_data_global_schema_graph.ttl')
    # column similarity thresholds. Columns having equal or higher similarities will have a relationship in the graph
    # label similarity threshold (column name)
    col_label_sim_threshold = 0.75
    # embedding similarity threshold (column values)
    col_embedding_sim_threshold = 0.75
    # boolean similarity threshold (similarity between boolean columns).
    col_boolean_sim_threshold = 0.75
