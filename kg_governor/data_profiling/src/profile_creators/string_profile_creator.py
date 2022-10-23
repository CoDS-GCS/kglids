import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow log level to FATAL

import pandas as pd
import torch

import chars2vec

from profile_creators.textual_profile_creator import TextualProfileCreator
from model.table import Table
from model.column_data_type import ColumnDataType
from column_embeddings.string_model import StringEmbeddingModel, StringScalingModel
from column_embeddings.utils import load_pretrained_model


class StringProfileCreator(TextualProfileCreator):

    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)

        # set the data type and load the embedding models
        self.data_type = ColumnDataType.STRING

        embedding_model_path = 'column_embeddings/pretrained_models/string/20221020165957_string_model_embedding_epoch_100.pt'
        scaling_model_path = 'column_embeddings/pretrained_models/string/20221020165957_string_model_scaling_epoch_100.pt'

        self.embedding_model = load_pretrained_model(StringEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(StringScalingModel, scaling_model_path)

    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        char_level_embed_model = chars2vec.load_model('eng_50')
        input_vector = char_level_embed_model.vectorize_words(self.column.dropna().tolist())
        input_tensor = torch.FloatTensor(input_vector).to(device)
        return input_tensor
