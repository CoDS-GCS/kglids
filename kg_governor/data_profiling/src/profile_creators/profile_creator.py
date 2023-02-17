from abc import ABC, abstractmethod

import pandas as pd
import torch

from model.column_data_type import ColumnDataType
from model.table import Table
from utils import generate_column_id, generate_table_id, generate_dataset_id
from column_embeddings.column_embeddings_utils import EMBEDDING_SIZE


class ProfileCreator(ABC):
    
    # Profile Creator Factory
    @staticmethod
    def get_profile_creator(column: pd.Series, data_type: ColumnDataType, table: Table):
        if data_type == ColumnDataType.INT:
            from profile_creators.int_profile_creator import IntProfileCreator
            return IntProfileCreator(column, table)
        elif data_type == ColumnDataType.FLOAT:
            from profile_creators.float_profile_creator import FloatProfileCreator
            return FloatProfileCreator(column, table)
        elif data_type == ColumnDataType.BOOLEAN:
            from profile_creators.boolean_profile_creator import BooleanProfileCreator
            return BooleanProfileCreator(column, table)
        elif data_type == ColumnDataType.DATE:
            from profile_creators.date_profile_creator import DateProfileCreator
            return DateProfileCreator(column, table)
        elif data_type == ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY:
            from profile_creators.named_entity_profile_creator import NamedEntityProfileCreator
            return NamedEntityProfileCreator(column, table)
        elif data_type == ColumnDataType.NATURAL_LANGUAGE_TEXT:
            from profile_creators.natural_language_text_profile_creator import NaturalLanguageTextProfileCreator
            return NaturalLanguageTextProfileCreator(column, table)
        elif data_type == ColumnDataType.STRING:
            from profile_creators.string_profile_creator import StringProfileCreator
            return StringProfileCreator(column, table)
        else:
            raise ValueError('Unknown column data type: ' + str(data_type))
    
    
    def __init__(self, column: pd.Series, table: Table):
        self.column = column
        self.table = table
        self.column_name = str(self.column.name)
        self.table_name = self.table.get_table_name()
        self.dataset_name = self.table.get_dataset_name()
        self.data_source = self.table.get_data_source()
        self.path = self.table.get_table_path()
        self.column_id = generate_column_id(self.data_source, self.dataset_name, self.table_name, str(self.column.name))
        self.table_id = generate_table_id(self.data_source, self.dataset_name, self.table_name)
        self.dataset_id = generate_dataset_id(self.data_source, self.dataset_name)
        self.data_type = None
        self.embedding_model = None
        self.scaling_model = None
        self.embedding = None
        self.embedding_scaling_factor = None
        
        # stats
        self.total_values_count = len(self.column)
        self.missing_values_count = len(self.column) - self.column.count().item()
        self.distinct_values_count = self.column.nunique()
    
    def _generate_embedding(self):
        if self.column.count():
            preprocessed_column = self._preprocess_column_for_embedding_model()
        
            with torch.inference_mode():
                self.embedding = self.embedding_model(preprocessed_column).mean(axis=0).tolist()
                self.embedding_scaling_factor = self.scaling_model(preprocessed_column).mean().item()
        else:
            # if the column is empty (nan values), set the embedding to zeros and the scaling factor to 1
            self.embedding = [0] * EMBEDDING_SIZE
            self.embedding_scaling_factor = 1

    @abstractmethod
    def create_profile(self):
        pass
    
    @abstractmethod
    def _calculate_stats(self):
        pass
    
    @abstractmethod
    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        pass
