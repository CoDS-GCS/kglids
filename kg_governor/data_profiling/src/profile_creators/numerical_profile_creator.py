import bitstring
import numpy as np
import pandas as pd
import torch

from profile_creators.profile_creator import ProfileCreator
from model.column_profile import ColumnProfile
from column_embeddings.numerical_model import NumericalEmbeddingModel, NumericalScalingModel
from column_embeddings.utils import load_pretrained_model
from model.table import Table
from model.column_data_type import ColumnDataType

class NumericalProfileCreator(ProfileCreator):

    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)

        # set the data type and load the embedding models
        self.data_type = ColumnDataType.NUMERICAL

        embedding_model_path = 'column_embeddings/pretrained_models/numerical/20221030142854_numerical_model_embedding_epoch_35.pt'
        scaling_model_path = 'column_embeddings/pretrained_models/numerical/20221030142854_numerical_model_scaling_epoch_35.pt'

        self.embedding_model = load_pretrained_model(NumericalEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(NumericalScalingModel, scaling_model_path)
    
    
    def create_profile(self):
        self._calculate_stats()
        self._generate_embedding()

        column_profile = ColumnProfile(column_id=self.column_id, dataset_name=self.dataset_name,
                                       dataset_id=self.dataset_id, path=self.path, table_name=self.table_name,
                                       table_id=self.table_id, column_name=self.column_name, data_source=self.data_source,
                                       data_type=self.data_type.value, total_values=self.total_values_count,
                                       distinct_values_count=self.distinct_values_count,
                                       missing_values_count=self.missing_values_count, min_value=self.min,
                                       max_value=self.max, mean=self.mean, median=self.median, iqr=self.iqr,
                                       embedding=self.embedding, embedding_scaling_factor=self.embedding_scaling_factor)
        return column_profile

    def _calculate_stats(self):
        summary = self.column.describe().dropna()
        self.mean = summary['mean'].item() if 'mean' in summary.index else None
        self.std = summary['std'].item() if 'std' in summary.index else None
        self.min = summary['min'].item() if 'min' in summary.index else None
        self.max = summary['max'].item() if 'max' in summary.index else None
        self.median = summary['50%'].item() if '50%' in summary.index else None
        self.iqr = (summary['75%'] - summary['25%']).item() if '75%' in summary.index else None

    
    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        if self.column.dtype.type in [np.bool_, np.int64, np.uint64]:
            bin_repr = [[int(j) for j in bitstring.Bits(int=int(min(value, 2**31-1)), length=32).bin]
                        for value in self.column.dropna().values]
        else:
            bin_repr = [[int(j) for j in bitstring.Bits(float=float(value), length=32).bin] 
                        for value in self.column.dropna().values]
        input_tensor = torch.FloatTensor(bin_repr).to(device)
        return input_tensor
