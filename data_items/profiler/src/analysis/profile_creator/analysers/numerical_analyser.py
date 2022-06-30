from analysis.profile_creator.analysers.i_analyser import IAnalyser
from analysis.utils import init_spark
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from deep_embeddings.numerical_model import load_embedding_model
import bitstring
import torch


DEVICE = 'cpu'
EMBEDDING_MODEL = load_embedding_model('deep_embeddings/embedding_model/20211123161253_numerical_embedding_model_epoch_4_3M_samples_gpu_cluster.pt')
# print('Deep embedding model loaded successfully!')


class NumericalAnalyser(IAnalyser):

    def __init__(self, df: DataFrame):
        self.spark = init_spark()
        self.profiles_info = {}
        self.df = df

    def analyse_columns(self):
        columns = self.df.columns
        if not columns:
            return
        summaries = self.__extract_summaries(columns)
        subtypes = self.__get_subtypes()
        num_distinct_values_counts_per_column_dict = self.__compute_distinct_values_counts(columns)
        missing_values_per_column_dict = self.__get_missing_values(columns)
        quartiles = self.__get_quantiles(columns)
        embeddings = self.__get_embeddings(columns)
        for col in columns:
            profile_info = {**{'type': subtypes[col]},
                            **{'count': summaries[col][0]},
                            **{'mean': summaries[col][1]},
                            **{'stddev': summaries[col][2]},
                            **{'min': summaries[col][3]},
                            **{'max': summaries[col][4]},
                            **{'distinct_values_count': num_distinct_values_counts_per_column_dict[col][0]},
                            **{'missing_values_count': missing_values_per_column_dict[col]},
                            **{'25%': quartiles['`' +col+'`' ][0]},
                            **{'50%': quartiles['`' +col+'`' ][1]},
                            **{'75%': quartiles['`' +col+'`' ][2]},
                            **{'deep_embedding': embeddings[col]}}
            self.profiles_info[col] = profile_info

    def __extract_summaries(self, columns: list):
        summary = self.df.describe()
        return summary.toPandas().to_dict()

    def __compute_distinct_values_counts(self, columns: list) -> dict:
        return self.df.agg(*(countDistinct(col('`' + c + '`')).alias(c) for c in columns)).toPandas().to_dict()

    def __get_missing_values(self, columns: list) -> dict:
        return self.df.select(*(sum(col('`' + c + '`').isNull().cast("int")).alias(c) for c in columns)) \
            .rdd \
            .map(lambda x: x.asDict()) \
            .collect()[0]

    def __get_quantiles(self, quantile_list) -> dict:
        quantile_list = ['`' + c + '`' for c in quantile_list]
        quartiles = [0.25, 0.5, 0.75]
        quartilesDF = self.spark.createDataFrame(
            zip(quartiles, *self.df.approxQuantile(quantile_list, quartiles, 0.03)),
            ["Percentile"] + quantile_list
        )
        return quartilesDF.toPandas().to_dict()
    
    def __get_subtypes(self) -> dict:
        numerical_cols = [f.name for f in self.df.schema.fields if not isinstance(f.dataType, StringType)]
        numerical_subtypes = {}
        for col in numerical_cols:
            distinct_vals = [i[0] for i in self.df.select(col).sample(0.1).distinct().collect()]
            if not distinct_vals:
                distinct_vals = [i[0] for i in self.df.select(col).distinct().collect()]    # if len(df) < 10
            
            # TODO: [Refactor] use more descriptive names for data types. Also read them from config/enum
            if distinct_vals == [0, 1] or [1, 0]:
                numerical_subtypes[col] = 'N_bool'
            elif distinct_vals == [int(i) for i in distinct_vals]:
                numerical_subtypes[col] = 'N_int'
            else:
                numerical_subtypes[col] = 'N_float'
        
        return numerical_subtypes

    def __get_embeddings(self, columns) -> dict:
        def compute_deep_embeddings(col):
            bin_repr = [[int(j) for j in bitstring.BitArray(float=float(i), length=32).bin] for i in col]
            bin_tensor = torch.FloatTensor(bin_repr).to(DEVICE)
            with torch.no_grad():
                embedding_tensor = EMBEDDING_MODEL(bin_tensor).mean(axis=0)
            return embedding_tensor.tolist()

        deep_embeddingsUDF = udf(lambda z: compute_deep_embeddings(z), ArrayType(FloatType()))
        cols = self.df.columns
        cols2 = ['`' + c + '`' for c in cols]
        df2 = self.df.select([collect_list(c) for c in cols2]).toDF(*cols2)
        df2 = df2.toDF(*cols)
        for col in cols:
            df2 = df2.withColumn(col, deep_embeddingsUDF('`' + col + '`'))
        deep_embeddings = {}
        d = df2.toPandas().to_dict()
        for c in cols:
            deep_embeddings[c] = d[c][0]
        return deep_embeddings
            
    def get_profiles_info(self):
        return self.profiles_info
