import os

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def init_spark():
    mem_gb = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 1024**3) - 1

    # TODO: [Refactor] have Spark configuration read from the global project config
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", f'{mem_gb}g')\
        .config('spark.local.dir', 'temp_spark/')\
        .config('spark.driver.maxResultSize', f'{mem_gb}g') \
        .getOrCreate()
    return spark

'''
def get_columns_in_rdd(df: DataFrame) -> RDD:
    return df.rdd.map(lambda row: row.asDict()) \
        .flatMap(lambda d: [(c, d[c]) for c in d.keys()]) \
        .map(lambda t: (t[0], '') if t[1] is None else t) \
        .groupByKey().map(lambda x: (x[0], list(x[1])))
'''


def get_columns(df: DataFrame) -> list:
    cols = []
    for c in df.columns:
        cols.append(('`' + c + '`', df.select('`' + c + '`').rdd.flatMap(lambda x: x).collect()))
    return cols
