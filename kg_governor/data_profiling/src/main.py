from datetime import datetime
import glob
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Tuple
import warnings
warnings.simplefilter('ignore')

from tqdm import tqdm
import pandas as pd
from pyspark import SparkConf, SparkContext

from fine_grained_type_detector import FineGrainedColumnTypeDetector
from profile_creators.profile_creator import ProfileCreator
from model.table import Table
from config import profiler_config


def main():
    start_time = datetime.now()
    print(datetime.now(), ': Initializing Spark')
    # initialize spark
    spark = SparkContext(conf=SparkConf().setMaster(f'local[*]')
                         .set('spark.driver.memory', f'{profiler_config.max_memory}g'))


    if os.path.exists(profiler_config.output_path):
        print(datetime.now(), ': Deleting existing column profiles in:', profiler_config.output_path)
        shutil.rmtree(profiler_config.output_path)

    os.makedirs(profiler_config.output_path, exist_ok=True)

    # get the list of columns and their associated tables
    print(datetime.now(), ': Creating tables, Getting columns')
    columns_and_tables = []
    for data_source in profiler_config.data_sources:
        for filename in glob.glob(os.path.join(data_source.path, '**/*' + data_source.file_type), recursive=True):
            if os.path.getsize(filename) > 0:   # if file is not empty
                table = Table(data_source=data_source.name,
                                    table_path=filename,
                                    dataset_name=Path(filename).resolve().parent.name)
                # read only the header
                header = pd.read_csv(table.get_table_path(), nrows=0, engine='python', encoding_errors='replace')
                columns_and_tables.extend([(col, table) for col in header.columns])

    columns_and_tables_rdd = spark.parallelize(columns_and_tables)
    
    # profile the columns with Spark.
    print(datetime.now(), f': Profiling {len(columns_and_tables)} columns')    
    columns_and_tables_rdd.map(column_worker).collect()

    print(datetime.now(), f': {len(columns_and_tables)} columns profiled and saved to {profiler_config.output_path}')
    print(datetime.now(), ': Total time to profile: ', datetime.now() - start_time)


def column_worker(column_name_and_table: Tuple[str, Table]):
    column_name, table = column_name_and_table
    # read the column from the table file
    column = pd.read_csv(table.get_table_path(), usecols=[column_name], squeeze=True,
                         na_values=[' ', '?'], engine='python', encoding_errors='replace')
    
    # infer the column data type
    column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column)
    
    # collect statistics, generate embeddings, and create the column profiles
    column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, table)
    column_profile = column_profile_creator.create_profile()
    
    # store the profile
    column_profile.save_profile(profiler_config.output_path)


if __name__ == '__main__':
    main()
