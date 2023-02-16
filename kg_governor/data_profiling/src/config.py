import os


class DataSource:
    def __init__(self, name, file_type, path):
        # a human-friendly name for the data source to distinguish it from the others. E.g. 'kaggle'
        self.name = name
        # path to the directory containing collection of datasets in this data source
        self.path = path
        # extension of the tables in this data source (usually 'csv')
        self.file_type = file_type
        
        
class ProfilerConfig:
    
    # list of data sources to process
    data_sources = [DataSource(name='santos',
                               path='/home/mossad/projects/kglids/storage/data_sources/dataset_storage/experiments/santos/benchmark/',
                               file_type='csv')]
    # data_sources = [DataSource(name='smaller_real',
    #                            path='/home/mossad/projects/kglids/storage/data_sources/dataset_storage/experiments/smallerReal/benchmark',
    #                            file_type='csv')]

    # directory to save the generated column profiles. 
    output_path = '/home/mossad/projects/kglids/storage/profiles/santos_profiles_fine_grained/'
    
    # whether to run Spark in local or cluster mode. 
    is_spark_local_mode = True
    # number of workers (processes) to use when profiling columns. Defaults to the number of threads.
    n_workers = os.cpu_count()
    # maximum memory in GB to be used by Spark
    max_memory = 25
    


profiler_config = ProfilerConfig()
