import warnings
warnings.simplefilter('ignore')
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow log level to FATAL


import spacy
import dateparser
import fasttext
fasttext.FastText.eprint = lambda *args,**kwargs: None
import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np

from model.column_data_type import ColumnDataType


class FineGrainedColumnTypeDetector:
    try:
        ner_model = spacy.load('en_core_web_sm')
    except:
        import subprocess
        subprocess.call('python -m spacy download en_core_web_sm'.split(), shell=False)
        ner_model = spacy.load('en_core_web_sm')
    
    fasttext_model = fasttext.load_model(str(Path(__file__).parent) + '/fasttext_embeddings/cc.en.50.bin')
    tokenizer = TweetTokenizer()

    @staticmethod
    def detect_column_data_type(column: pd.Series):
        if column.dtype.type == np.bool_:
            return ColumnDataType.BOOLEAN
        
        elif column.dtype.type in [np.int64, np.uint64]:
            if set(column.unique()) == {0, 1}:
                return ColumnDataType.BOOLEAN
            return ColumnDataType.INT
        
        elif column.dtype.type == np.float64:
            return ColumnDataType.FLOAT
        
        else:
            SAMPLE_SIZE = 1000
            sample = column.sample(min(len(column), SAMPLE_SIZE)).dropna()
            
            if FineGrainedColumnTypeDetector.__is_date(sample):
                return ColumnDataType.DATE
            
            elif FineGrainedColumnTypeDetector.__is_natural_language(sample):
                if FineGrainedColumnTypeDetector.__is_named_entity(sample):
                    return ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY
                
                return ColumnDataType.NATURAL_LANGUAGE_TEXT
            
            return ColumnDataType.STRING
    
    
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
    
    @staticmethod
    def __is_named_entity(column: pd.Series):
        num_named_entity_values = 0
        for value in column.values:
            tokens = FineGrainedColumnTypeDetector.ner_model(value)
            non_puncts = len([token for token in tokens if not token.is_punct and not token.is_space])
            if len(tokens.ents) == non_puncts:
                num_named_entity_values += 1
                if num_named_entity_values > 0.5 * len(column):
                    return True
        return False
    
    @staticmethod
    def __is_date(column: pd.Series):
        num_date_values = 0
        for value in column.values:
            # the value is a date if it is short enough and is parsed by the dateparser
            if len(value) < 50 and dateparser.parse(value, locales=['en-CA'], languages=['en'],
                                                    settings={'STRICT_PARSING': True}):
                num_date_values += 1
                if num_date_values > 0.5 * len(column):
                    return True
        return False
