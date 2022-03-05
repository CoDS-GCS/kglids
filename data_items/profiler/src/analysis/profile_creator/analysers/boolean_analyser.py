                                                            
from analysis.profile_creator.analysers.i_analyser import IAnalyser
from analysis.utils import init_spark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, countDistinct, sum


class BooleanAnalyser(IAnalyser):

    def __init__(self, df: DataFrame):
        self.spark = init_spark()
        self.profiles_info = {}
        self.df = df

    def analyse_columns(self):
        columns = self.df.columns
        if not columns:
            return
        num_distinct_values_counts_per_column_dict = self.__compute_distinct_values_counts(columns)
        missing_values_per_column_dict = self.__get_missing_values(columns)
        for c in columns:
            profile_info = {**{'count': self.df.select(col('`' + c + '`')).count()},
                            **{'distinct_values_count': num_distinct_values_counts_per_column_dict[c][0]},
                            **{'missing_values_count': missing_values_per_column_dict[c]}}
                           
            self.profiles_info[c] = profile_info


    def __compute_distinct_values_counts(self, columns: list) -> dict:
        return self.df.agg(*(countDistinct(col('`' + c + '`')).alias(c) for c in columns)).toPandas().to_dict()

    def __get_missing_values(self, columns: list) -> dict:
        return self.df.select(*(sum(col('`' + c + '`').isNull().cast("int")).alias(c) for c in columns)) \
            .rdd \
            .map(lambda x: x.asDict()) \
            .collect()[0]
     
    def get_profiles_info(self):
        return self.profiles_info
