from enum import Enum

from allennlp.predictors.predictor import Predictor
import dateparser
import fasttext
import pandas as pd
import numpy as np


class ColumnDataType(Enum):
    INT = 'int'
    FLOAT = 'float'
    BOOLEAN = 'boolean'
    DATE = 'date'
    NATURAL_LANGUAGE_NAMED_ENTITY = 'named_entity'
    NATURAL_LANGUAGE_TEXT = 'text'
    STRING = 'string'


class FineGrainedColumnTypeDetector:
    
    ner_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
    fasttext_model = fasttext.load_model('fasttext_embeddings/cc.en.50.bin')

    @staticmethod
    def detect_column_data_type(column: pd.Series):
        # TODO: before doing this: do df.convert_dtypes(), df.to_numeric()
        if column.dtype == np.bool_:
            return ColumnDataType.BOOLEAN
        
        elif column.dtype == np.int64:
            if column.isin([0, 1]):
                return ColumnDataType.BOOLEAN
            return ColumnDataType.INT
        
        elif column.dtype == np.float64:
            return ColumnDataType.FLOAT
        
        else:
            sample = column.sample(min(len(column), 1000))
            
            if FineGrainedColumnTypeDetector.__is_natural_language(sample):
                if FineGrainedColumnTypeDetector.__is_named_entity(sample):
                    return ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY
                
                return ColumnDataType.NATURAL_LANGUAGE_TEXT
            
            elif FineGrainedColumnTypeDetector.__is_date(sample):
                return ColumnDataType.DATE
            
            return ColumnDataType.STRING
    
    
    @staticmethod
    def __is_natural_language(column: pd.Series):
        return column.apply(lambda x: all([FineGrainedColumnTypeDetector.fasttext_model.get_word_id(word) != -1 for word in x.split()])).sum() > 0.5 * len(column)
    
    @staticmethod
    def __is_named_entity(column: pd.Series):
        return column.apply(lambda x: 'O' not in FineGrainedColumnTypeDetector.ner_model.predict(x)['tags']).sum() > 0.5 * len(column)
    
    @staticmethod
    def __is_date(column: pd.Series):
        return column.apply(lambda x: dateparser.parse(x, locales=['en-CA'], languages=['en'])).sum() > 0.5 * len(column)
    
    