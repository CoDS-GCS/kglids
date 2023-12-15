from torch import FloatTensor
import torch
import bitstring
import numpy as np
import pandas as pd
import dateparser
import compress_fasttext
from nltk.tokenize import TweetTokenizer
from gnn_applications.OnDemandDataPrep.utils.column_embeddings import load_numeric_embedding_model, load_string_embedding_model, \
    load_nl_embedding_model
from gnn_applications.OnDemandDataPrep.utils.fine_grained_type_detector import FineGrainedColumnTypeDetector
from gnn_applications.OnDemandDataPrep.utils.column_data_type import ColumnDataType

class EmbeddingCreator:
    def __init__(self):
        # load CoLR Embedding models
        self.int_embedding_model = load_numeric_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/int/20230123140419_int_model_embedding_epoch_100.pt')
        self.float_embedding_model = load_numeric_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/float/20230124151732_float_model_embedding_epoch_89.pt')
        self.date_embedding_model = load_numeric_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/date/20230113111008_date_model_embedding_epoch_100.pt')
        self.named_entity_embedding_model = load_nl_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/named_entity/20230125181821_named_entity_model_embedding_epoch_34.pt')
        self.natural_language_text_embedding_model = load_nl_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/natural_language_text/20230113132355_natural_language_text_model_embedding_epoch_94.pt')
        self.string_embedding_model = load_string_embedding_model(
            path='kg_governor/data_profiling/src/column_embeddings/pretrained_models/string/20230111150300_string_model_embedding_epoch_100.pt')

    def new_emb(self, entity_df: pd.DataFrame, operation):
        def preprocess_date(column):
            non_missing = column.dropna()
            if len(non_missing) > 1000:
                sample = non_missing.sample(int(0.1 * len(non_missing)))
            else:
                sample = non_missing.sample(min(len(non_missing), 1000))
            dates = sample.apply(lambda x: dateparser.parse(x, locales=['en-CA'], languages=['en']))
            timestamps = dates.dropna().apply(lambda x: x.timestamp())
            bin_repr = [[int(j) for j in bitstring.BitArray(float=float(i), length=32).bin] for i in timestamps]
            input_tensor = FloatTensor(bin_repr).to('cpu')
            return input_tensor

        def preprocess_num(column):
            non_missing = column.dropna()
            if len(non_missing) > 1000:
                sample = non_missing.sample(int(0.1 * len(non_missing)))
            else:
                sample = non_missing.sample(min(len(non_missing), 1000))
            bin_repr = [[int(j) for j in bitstring.BitArray(float=float(i), length=32).bin]
                        for i in sample.values]
            input_tensor = FloatTensor(bin_repr).to('cpu')
            return input_tensor

        def preprocess_nl(column):
            non_missing = column.dropna()
            if len(non_missing) > 1000:
                sample = non_missing.sample(int(0.1 * len(non_missing)))
            else:
                sample = non_missing.sample(min(len(non_missing), 1000))
            fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
                'OnDemandDataPrep/utils/ft_cc.en.50.bin')
            tokenizer = TweetTokenizer()

            input_values = []
            for text in sample.values:
                fasttext_words = [word for word in tokenizer.tokenize(text) if fasttext_model.has_index_for(word)]

                if fasttext_words:
                    input_values.append(
                        np.average([fasttext_model.get_vector(word) for word in fasttext_words], axis=0))
            input_tensor = FloatTensor(input_values).to('cpu')
            if input_tensor.numel() == 0:
                input_tensor = torch.zeros(0, 50, dtype=torch.float32)

            return input_tensor

        embedding = {}
        embedding_300 = {}
        cleaning_embedding = {}
        numeric_column_embeddings = {}
        categorical_column_embeddings = {}
        for column in entity_df.columns:
            column_type = FineGrainedColumnTypeDetector.detect_column_data_type(entity_df[column])
            if column_type is ColumnDataType.DATE:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((np.zeros(600), self.date_embedding_model(
                        preprocess_date(entity_df[column])).mean(axis=0).tolist(), np.zeros(900)))
                    embedding_300[column] = self.date_embedding_model(preprocess_date(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                categorical_column_embeddings[column] = embedding[column]
            elif column_type is ColumnDataType.INT:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((self.int_embedding_model(preprocess_num(entity_df[column])).mean(
                                                           axis=0).tolist(), np.zeros(1500)))
                    embedding_300[column] = self.int_embedding_model(preprocess_num(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                numeric_column_embeddings[column] = embedding[column]
            elif column_type is ColumnDataType.FLOAT:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((np.zeros(300), self.float_embedding_model(
                        preprocess_num(entity_df[column])).mean(axis=0).tolist(), np.zeros(1200)))
                    embedding_300[column] = self.float_embedding_model(preprocess_num(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                numeric_column_embeddings[column] = embedding[column]

            elif column_type is ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((np.zeros(900), self.named_entity_embedding_model(
                        preprocess_nl(entity_df[column])).mean(axis=0).tolist(), np.zeros(600)))
                    embedding_300[column] = self.named_entity_embedding_model(preprocess_nl(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                categorical_column_embeddings[column] = embedding[column]
            elif column_type is ColumnDataType.NATURAL_LANGUAGE_TEXT:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((np.zeros(1200), self.natural_language_text_embedding_model(
                        preprocess_nl(entity_df[column])).mean(axis=0).tolist(), np.zeros(300)))
                    embedding_300[column] = self.natural_language_text_embedding_model(preprocess_nl(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                categorical_column_embeddings[column] = embedding[column]
            elif column_type is ColumnDataType.STRING:
                if entity_df[column].notna().any():
                    embedding[column] = np.concatenate((np.zeros(1500), self.string_embedding_model(
                        preprocess_nl(entity_df[column])).mean(axis=0).tolist()))  # NIKI replaced preprocess_nl
                    embedding_300[column] = self.string_embedding_model(preprocess_nl(entity_df[column])).mean(axis=0).tolist()
                else:
                    embedding[column] = np.zeros(1800)
                    embedding_300[column] = np.zeros(300)
                if entity_df[column].isna().sum().sum() > 0:
                    cleaning_embedding[column] = embedding[column]
                categorical_column_embeddings[column] = embedding[column]
            elif column_type is ColumnDataType.BOOLEAN:
                embedding[column] = np.zeros(1800)
                embedding_300[column] = np.zeros(300)
            else:
                print('Non-identified type')
        if operation == 'scaling-table':
            embedding_arrays = list(numeric_column_embeddings.values())  # ONLY NUMERICAL
            embedding_average = np.mean(embedding_arrays, axis=0)
            return embedding_average
        if operation == 'cleaning-table':
            embedding_arrays = list(cleaning_embedding.values())  # ONLY NUMERICAL
            embedding_average = np.mean(embedding_arrays, axis=0)
            return embedding_average
        if operation == 'scaling-column' or operation == 'cleaning-column':
            df = pd.DataFrame.from_dict(embedding, orient='index')
            return df
        if operation == 'unary-column' :
            df = pd.DataFrame.from_dict(embedding_300, orient='index')
            return df
