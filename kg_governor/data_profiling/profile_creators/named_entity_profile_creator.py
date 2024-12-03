import os

import numpy as np
import pandas as pd
import torch
import fasttext
fasttext.FastText.eprint = lambda *args,**kwargs: None
from nltk.tokenize import TweetTokenizer

from kg_governor.data_profiling.profile_creators.textual_profile_creator import TextualProfileCreator
from kg_governor.data_profiling.model.table import Table
from kg_governor.data_profiling.model.column_data_type import ColumnDataType
from kg_governor.data_profiling.column_embeddings.natural_language_model import NaturalLanguageEmbeddingModel, NaturalLanguageScalingModel
from kg_governor.data_profiling.column_embeddings.column_embeddings_utils import load_pretrained_model
from kglids_config import KGLiDSConfig


class NamedEntityProfileCreator(TextualProfileCreator):

    def __init__(self, column: pd.Series, table: Table, fasttext_model: fasttext.FastText):
        super().__init__(column, table)

        # set the data type and load the embedding models
        self.data_type = ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY
        self.fasttext_model = fasttext_model

        embedding_model_path = os.path.join(KGLiDSConfig.base_dir, 'kg_governor/data_profiling/column_embeddings/pretrained_models/named_entity/20230125181821_named_entity_model_embedding_epoch_34.pt')
        scaling_model_path = os.path.join(KGLiDSConfig.base_dir, 'kg_governor/data_profiling/column_embeddings/pretrained_models/named_entity/20230125181821_named_entity_model_scaling_epoch_34.pt')

        self.embedding_model = load_pretrained_model(NaturalLanguageEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(NaturalLanguageScalingModel, scaling_model_path)


    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        non_missing = self.column.dropna()
        if len(non_missing) > 1000:
            sample = non_missing.sample(int(0.1 * len(non_missing)))
        else:
            sample = non_missing.sample(min(len(non_missing), 1000))

        tokenizer = TweetTokenizer()

        input_values = []
        for text in sample.values:
            fasttext_words = [word for word in tokenizer.tokenize(text) if self.fasttext_model.get_word_id(word) != -1]
            if fasttext_words:
                input_values.append(np.average([self.fasttext_model.get_word_vector(word) for word in fasttext_words],
                                               axis=0))
        input_tensor = torch.FloatTensor(input_values).to(device)
        return input_tensor
