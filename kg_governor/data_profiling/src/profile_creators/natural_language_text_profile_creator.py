from pathlib import Path

import numpy as np
import pandas as pd
import torch
import fasttext
fasttext.FastText.eprint = lambda *args,**kwargs: None
from nltk.tokenize import TweetTokenizer

from profile_creators.textual_profile_creator import TextualProfileCreator
from model.table import Table
from model.column_data_type import ColumnDataType
from column_embeddings.natural_language_model import NaturalLanguageEmbeddingModel, NaturalLanguageScalingModel
from column_embeddings.utils import load_pretrained_model


class NaturalLanguageTextProfileCreator(TextualProfileCreator):

    def __init__(self, column: pd.Series, table: Table):
        super().__init__(column, table)

        # set the data type and load the embedding models
        self.data_type = ColumnDataType.NATURAL_LANGUAGE_TEXT

        embedding_model_path = 'column_embeddings/pretrained_models/natural_language/20221020165938_natural_language_model_embedding_epoch_14.pt'
        scaling_model_path = 'column_embeddings/pretrained_models/natural_language/20221020165938_natural_language_model_scaling_epoch_14.pt'

        self.embedding_model = load_pretrained_model(NaturalLanguageEmbeddingModel, embedding_model_path)
        self.scaling_model = load_pretrained_model(NaturalLanguageScalingModel, scaling_model_path)

    def _preprocess_column_for_embedding_model(self, device='cpu') -> torch.tensor:
        fasttext_path = str(Path(__file__).parent.parent) + '/fasttext_embeddings/cc.en.50.bin'
        fasttext_model = fasttext.load_model(fasttext_path)
        tokenizer = TweetTokenizer()

        input_values = []
        for text in self.column.dropna().values:
            fasttext_words = [word for word in tokenizer.tokenize(text) if fasttext_model.get_word_id(word) != -1]
            if fasttext_words:
                input_values.append(np.average([fasttext_model.get_word_vector(word) for word in fasttext_words],
                                               axis=0))
        input_tensor = torch.FloatTensor(input_values).to(device)
        return input_tensor
