from argparse import ArgumentParser
from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
from typing import Tuple
import warnings
warnings.simplefilter('ignore')
import hashlib

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from fine_grained_type_detector import FineGrainedColumnTypeDetector
from profile_creators.profile_creator import ProfileCreator
from model.table import Table
from config import profiler_config, DataSource
from utils import generate_column_id


def main():
    parser = ArgumentParser()
    parser.add_argument('--data-source-name', type=str, default=None)
    parser.add_argument('--data-source-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    args = parser.parse_args()
    if args.data_source_name and args.data_source_path:
        extra_source = DataSource(name=args.data_source_name, path=args.data_source_path)
        profiler_config.data_sources.append(extra_source)
    if args.output_path:
        profiler_config.output_path = args.output_path

    start_time = datetime.now()
    print(datetime.now(), ': Initializing Spark')

    # initialize spark
    if profiler_config.is_spark_local_mode:
        spark = SparkContext(conf=SparkConf().setMaster(f'local[{profiler_config.n_workers}]')
                             .set('spark.driver.memory', f'{profiler_config.max_memory//2}g'))
    else:

        spark = SparkSession.builder.appName("KGLiDSProfiler").getOrCreate().sparkContext
        # add python dependencies
        for pyfile in glob.glob('./**/*.py', recursive=True):
            spark.addPyFile(pyfile)
        # add embedding model files
        for embedding_file in glob.glob('./column_embeddings/pretrained_models/**/*.pt', recursive=True):
            spark.addFile(embedding_file)
        # add fasttext embeddings file
        spark.addFile('./fasttext_embeddings/cc.en.50.bin')

    if os.path.exists(profiler_config.output_path):
        print(datetime.now(), ': Deleting existing column profiles in:', profiler_config.output_path)
        shutil.rmtree(profiler_config.output_path)

    os.makedirs(profiler_config.output_path, exist_ok=True)

    # get the list of columns and their associated tables
    print(datetime.now(), ': Creating tables, Getting columns')
    columns_and_tables = []
    for data_source in profiler_config.data_sources:
        for filename in glob.glob(os.path.join(data_source.path, '**/*.' + data_source.file_type), recursive=True):
            if os.path.isfile(filename) and os.path.getsize(filename) > 0:   # if not an empty file
                dataset_base_dir = Path(filename).resolve()
                while dataset_base_dir.parent != Path(data_source.path).resolve():
                    dataset_base_dir = dataset_base_dir.parent
                table = Table(data_source=data_source.name,
                              table_path=filename,
                              dataset_name=dataset_base_dir.name)
                # read only the header
                try:
                    header = pd.read_csv(table.get_table_path(), nrows=0, engine='python', encoding_errors='replace')
                except:
                    continue
                columns_and_tables.extend([(col, table) for col in header.columns])

    columns_and_tables_rdd = spark.parallelize(columns_and_tables, len(columns_and_tables)//10)

    # profile the columns with Spark.
    print(datetime.now(), f': Profiling {len(columns_and_tables)} columns')
    columns_and_tables_rdd.map(column_worker).collect()

    print(datetime.now(), f': {len(columns_and_tables)} columns profiled and saved to {profiler_config.output_path}')
    print(datetime.now(), ': Total time to profile: ', datetime.now() - start_time)


def column_worker(column_name_and_table: Tuple[str, Table]):
    column_name, table = column_name_and_table


    # read the column from the table file. Use the Python engine if there are issues reading the file
    try:
        try:
            column = pd.read_csv(table.get_table_path(), usecols=[column_name], squeeze=True, na_values=[' ', '?', '-'])
        except:
            column = pd.read_csv(table.get_table_path(), usecols=[column_name], squeeze=True, na_values=[' ', '?', '-'],
                                 engine='python', encoding_errors='replace')

        column = pd.to_numeric(column, errors='ignore')
        column = column.convert_dtypes()
        column = column.astype(str) if column.dtype == object else column

        # infer the column data type
        column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column)

        # collect statistics, generate embeddings, and create the column profiles
        column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, table)
        column_profile = column_profile_creator.create_profile()

        # store the profile
        column_profile.save_profile(profiler_config.output_path)
    except:
        print(f'Warning: Skipping non-parse-able column: {column_name} in table: {table.get_table_path()}')
        return


if __name__ == '__main__':
    main()
