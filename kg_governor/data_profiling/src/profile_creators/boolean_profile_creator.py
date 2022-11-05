import pandas as pd

from profile_creators.profile_creator import ProfileCreator
from model.table import Table
from model.column_profile import ColumnProfile
from model.column_data_type import ColumnDataType


class BooleanProfileCreator(ProfileCreator):
    
    true_ratio = None
    
    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)
        
        self.data_type = ColumnDataType.BOOLEAN
        
    def create_profile(self):
        self._calculate_stats()

        column_profile = ColumnProfile(column_id=self.column_id, dataset_name=self.dataset_name,
                                       dataset_id=self.dataset_id, path=self.path, table_name=self.table_name,
                                       table_id=self.table_id, column_name=self.column_name,
                                       data_source=self.data_source, data_type=self.data_type.value,
                                       total_values=self.total_values_count,
                                       distinct_values_count=self.distinct_values_count,
                                       missing_values_count=self.missing_values_count, true_ratio=self.true_ratio)
        return column_profile

    def _calculate_stats(self):
        self.true_ratio = self.column.sum() / len(self.column)
    
    def _preprocess_column_for_embedding_model(self, device='cpu'):
        pass
