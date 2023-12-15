import warnings
warnings.simplefilter('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow log level to FATAL
import nltk
from nltk.tag import StanfordNERTagger
import dateparser
import fasttext
fasttext.FastText.eprint = lambda *args,**kwargs: None
import compress_fasttext
import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np

from gnn_applications.OnDemandDataPrep.utils.column_data_type import ColumnDataType


class FineGrainedColumnTypeDetector:
    stanford_ner_model = "OnDemandDataPrep/utils/english.muc.7class.distsim.crf.ser.gz"
    stanford_jar_file = "OnDemandDataPrep/utils/stanford-ner.jar"
    ner_tagger = StanfordNERTagger(stanford_ner_model, stanford_jar_file)
    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('OnDemandDataPrep/utils/ft_cc.en.50.bin')
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
            num_tokens_in_fasttext = sum([token in FineGrainedColumnTypeDetector.fasttext_model for token in tokens])

            if num_tokens_in_fasttext > 0.5 * len(tokens):
                num_natural_language_values += 1
                if num_natural_language_values > 0.5 * len(column):
                    return True
        return False

    @staticmethod
    def __is_named_entity(column: pd.Series):
        num_named_entity_values = 0
        for value in column.values:
            # print('value:',value)
            tokens = nltk.word_tokenize(value)
            tags = FineGrainedColumnTypeDetector.ner_tagger.tag(tokens)
            # print('tags:',tags)
    
            non_puncts = len([token for token, tag in tags if tag != 'O'])
            if len([tag for _, tag in tags if tag != 'O']) == non_puncts:
                num_named_entity_values += 1
                if num_named_entity_values > 0.5 * len(tokens):
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
