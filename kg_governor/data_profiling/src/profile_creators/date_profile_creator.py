import bitstring
import dateparser
import pandas as pd
import torch

from profile_creators.profile_creator import ProfileCreator
from model.column_profile import ColumnProfile
from model.table import Table
from model.column_data_type import ColumnDataType
from column_embeddings.numerical_model import NumericalEmbeddingModel, NumericalScalingModel
from column_embeddings.utils import load_pretrained_model

class DateProfileCreator(ProfileCreator):
    
    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)
        
        self.data_type = ColumnDataType.DATE

        embedding_model_path = 'column_embeddings/pretrained_models/numerical/20221030000534_numerical_model_embedding_epoch_35.pt'
        scaling_model_path = 'column_embeddings/pretrained_models/numerical/20221030000534_numerical_model_scaling_epoch_35.pt'

        self.embedding_model = load_pretrained_model(NumericalEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(NumericalScalingModel, scaling_model_path)
    
    
    def create_profile(self):
        self._generate_embedding()

        column_profile = ColumnProfile(column_id=self.column_id, dataset_name=self.dataset_name,
                                       dataset_id=self.dataset_id, path=self.path, table_name=self.table_name,
                                       table_id=self.table_id, column_name=self.column_name, data_source=self.data_source,
                                       data_type=self.data_type.value, total_values=self.total_values_count,
                                       distinct_values_count=self.distinct_values_count,
                                       missing_values_count=self.missing_values_count,
                                       embedding=self.embedding, embedding_scaling_factor=self.embedding_scaling_factor)
        return column_profile
    
    
    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        dates = self.column.dropna().apply(lambda x: dateparser.parse(x, locales=['en-CA'], languages=['en']))
        timestamps = dates.dropna().apply(lambda x: x.timestamp())
        bin_repr = [[int(j) for j in bitstring.BitArray(float=float(i), length=32).bin] for i in timestamps]
        input_tensor = torch.FloatTensor(bin_repr).to(device)
        return input_tensor
    
    def _calculate_stats(self):
        pass