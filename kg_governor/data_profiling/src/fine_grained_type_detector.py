import warnings
warnings.simplefilter('ignore')
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow log level to FATAL



import fasttext
fasttext.FastText.eprint = lambda *args,**kwargs: None
import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np

from model.column_data_type import ColumnDataType


class FineGrainedColumnTypeDetector:
    
    fasttext_model = fasttext.load_model(str(Path(__file__).parent) + '/fasttext_embeddings/cc.en.50.bin')
    tokenizer = TweetTokenizer()

    @staticmethod
    def detect_column_data_type(column: pd.Series):

        if column.dtype.type in [np.bool_, np.int64, np.uint64, np.float64]:
            return ColumnDataType.NUMERICAL
        
        else:
            SAMPLE_SIZE = 1000
            sample = column.sample(min(len(column), SAMPLE_SIZE)).dropna()
            
            if FineGrainedColumnTypeDetector.__is_natural_language(sample):
                return ColumnDataType.NATURAL_LANGUAGE
            
            return ColumnDataType.GENERAL_STRING
    
    
    @staticmethod
    def __is_natural_language(column: pd.Series):
        num_natural_language_values = 0
        for value in column.values:
            tokens = FineGrainedColumnTypeDetector.tokenizer.tokenize(value)
            num_tokens_in_fasttext = sum([FineGrainedColumnTypeDetector.fasttext_model.get_word_id(token) != -1
                                          for token in tokens])
            if num_tokens_in_fasttext > 0.5 * len(tokens):
                num_natural_language_values += 1
                if num_natural_language_values > 0.5 * len(column):
                    return True
        return False
    
