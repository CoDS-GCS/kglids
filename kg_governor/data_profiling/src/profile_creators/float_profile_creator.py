import pandas as pd

from profile_creators.numerical_profile_creator import NumericalProfileCreator
from model.table import Table
from model.column_data_type import ColumnDataType
from column_embeddings.numerical_model import NumericalEmbeddingModel, NumericalScalingModel
from column_embeddings.column_embeddings_utils import load_pretrained_model


class FloatProfileCreator(NumericalProfileCreator):

    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)

        # set the data type and load the embedding models
        self.data_type = ColumnDataType.FLOAT

        embedding_model_path = 'column_embeddings/pretrained_models/float/20230124151732_float_model_embedding_epoch_89.pt'
        scaling_model_path = 'column_embeddings/pretrained_models/float/20230124151732_float_model_scaling_epoch_89.pt'

        self.embedding_model = load_pretrained_model(NumericalEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(NumericalScalingModel, scaling_model_path)