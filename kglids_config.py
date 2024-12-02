# Contains all project configs
import os
from pathlib import Path

class KGLiDSConfig:
    ####### Data source configs #######
    data_source = 'kaggle_small'
    data_source_path = '/home/mossad/projects/kglids/storage/data_sources/kaggle_small'


    ####### General configs #######
    base_dir = str(Path.resolve(Path(__file__).parent))
    graphdb_repo_name = data_source
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
    replace_existing_profiles = True


    ####### Pipeline abstraction configs #######


    ####### KG construction configs #######


