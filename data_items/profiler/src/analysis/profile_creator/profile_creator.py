from analysis.profile_creator.analysers.numerical_analyser import NumericalAnalyser
from analysis.profile_creator.analysers.textual_analyser import TextualAnalyser
from analysis.profile_creator.analysers.boolean_analyser import BooleanAnalyser
from data.column_profile import ColumnProfile
from data.tables.i_table import ITable
from pyspark.sql import DataFrame
from utils import generate_id, generate_table_id, generate_dataset_id


class ProfileCreator:

    def __init__(self, table: ITable):
        self.table = table

    def create_numerical_profiles(self, numerical_cols_df: DataFrame):
        numerical_analyser = NumericalAnalyser(numerical_cols_df)
        numerical_analyser.analyse_columns()
        profiles_info = numerical_analyser.get_profiles_info()
        datasource = self.table.get_datasource()
        dataset_name = self.table.get_dataset_name()
        dataset_id = generate_dataset_id(datasource, dataset_name)
        table_name = self.table.get_table_name()
        table_id = generate_table_id(datasource, dataset_name, table_name)
        path = self.table.get_table_path()
        origin = self.table.get_origin()
        for column_name in numerical_cols_df.columns:
            column_id = generate_id(datasource, dataset_name, table_name, column_name)
            profile_info = profiles_info[column_name]
            profile = ColumnProfile(column_id, origin, dataset_name, dataset_id, path, table_name, table_id, column_name,
                                    datasource, 
                                  'N',  # TODO: [Refactor] use more descriptive names for data types
                                    float(profile_info['count']),
                                    float(profile_info['distinct_values_count']),
                                    float(profile_info['missing_values_count']),
                                    float(profile_info['min']), float(profile_info['max']),
                                    float(profile_info['mean']), float(profile_info['50%']),
                                    float(profile_info['75%']) - float(profile_info['25%']), [],
                                    profile_info['deep_embeddings'])
            yield profile

    def create_textual_profiles(self, textual_cols_df: DataFrame):
        textual_analyser = TextualAnalyser(textual_cols_df)
        textual_analyser.analyse_columns()
        profiles_info = textual_analyser.get_profiles_info()
        datasource = self.table.get_datasource()
        dataset_name = self.table.get_dataset_name()
        dataset_id = generate_dataset_id(datasource, dataset_name)
        table_name = self.table.get_table_name()
        table_id = generate_table_id(datasource, dataset_name, table_name)
        path = self.table.get_table_path()
        origin = self.table.get_origin()
        for column_name in textual_cols_df.columns:
            column_id = generate_id(datasource, dataset_name, table_name, column_name)
            profile_info = profiles_info[column_name]
            profile = ColumnProfile(column_id, origin, dataset_name, dataset_id, path, table_name, table_id, column_name,
                                    datasource,
                                    str(profile_info['string_subtype']),  # TODO: [Refactoring] use more descriptive names for data types
                                    float(profile_info['count']),
                                    float(profile_info['distinct_values_count']),
                                    float(profile_info['missing_values_count']),
                                    -1, -1, -1, -1, -1,
                                    profile_info['minhash'], [])
            yield profile

    def create_boolean_profiles(self, boolean_cols_df: DataFrame):
        boolean_analyser = BooleanAnalyser(boolean_cols_df)
        boolean_analyser.analyse_columns()
        profiles_info = boolean_analyser.get_profiles_info()
        datasource = self.table.get_datasource()
        dataset_name = self.table.get_dataset_name()
        dataset_id = generate_dataset_id(datasource, dataset_name)
        table_name = self.table.get_table_name()
        table_id = generate_table_id(datasource, dataset_name, table_name)
        path = self.table.get_table_path()
        origin = self.table.get_origin()
        for column_name in boolean_cols_df.columns:
            column_id = generate_id(datasource, dataset_name, table_name, column_name)
            profile_info = profiles_info[column_name]
            profile = ColumnProfile(column_id, origin, dataset_name, dataset_id, path, table_name, table_id, column_name,
                                    datasource, 
                                  'B',  # TODO: [Refactor] use more descriptive names for data types
                                    float(profile_info['count']),
                                    float(profile_info['distinct_values_count']),
                                    float(profile_info['missing_values_count']),
                                    -1, -1, -1, -1, -1, [], [])
            yield profile
