import os
import random
import json
import string
from .column_data_type import ColumnDataType


class ColumnProfile:
    def __init__(self, column_id: str, dataset_name: str, dataset_id: str, path: str, table_name: str,
                 table_id: str, column_name: str, data_source: str, data_type: str,
                 total_values: float, distinct_values_count: float, missing_values_count: float,
                 true_ratio: float = None, min_value: float = None, max_value: float = None,
                 mean: float = None, median: float = None, iqr: float = None,
                 embedding: list = None, embedding_scaling_factor: float = None):
        self.column_id = column_id
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.path = path
        self.table_name = table_name
        self.table_id = table_id
        self.data_source = data_source
        self.column_name = column_name
        self.data_type = data_type
        self.total_values = total_values
        self.distinct_values_count = distinct_values_count
        self.missing_values_count = missing_values_count
        self.true_ratio = true_ratio
        self.max_value = max_value
        self.min_value = min_value
        self.mean = mean
        self.median = median
        self.iqr = iqr
        self.embedding = embedding
        self.embedding_scaling_factor = embedding_scaling_factor

    def to_dict(self):
        profile_dict = {'column_id': self.get_column_id(),
                        'dataset_name': self.get_dataset_name(),
                        'dataset_id': self.get_dataset_id(),
                        'path': self.get_path(),
                        'table_name': self.get_table_name(),
                        'table_id': self.get_table_id(),
                        'column_name': self.get_column_name(),
                        'data_source': self.get_data_source(),
                        'data_type': self.get_data_type(), 
                        'total_values_count': self.get_total_values_count(),
                        'distinct_values_count': self.get_distinct_values_count(),
                        'missing_values_count': self.get_missing_values_count(),
                        'true_ratio': self.get_true_ratio(),
                        'min_value': self.get_min_value(), 
                        'max_value': self.get_max_value(), 
                        'mean': self.get_mean(),
                        'median': self.get_median(), 
                        'iqr': self.get_iqr(),
                        'embedding': self.get_embedding(),
                        'embedding_scaling_factor': self.get_embedding_scaling_factor()}
        return profile_dict
    
    def save_profile(self, column_profile_base_dir):
        profile_save_path = os.path.join(column_profile_base_dir, self.get_data_type())
        os.makedirs(profile_save_path, exist_ok=True)
        # random generated name of length 10 to avoid synchronization between threads and profile name collision
        profile_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))  
        with open(os.path.join(profile_save_path, f'{profile_name}.json'), 'w') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
    
    
    @staticmethod
    def load_profile(column_profile_path):
        with open(column_profile_path) as f:
            profile_dict = json.load(f)
        profile = ColumnProfile(column_id=profile_dict.get('column_id'),
                                dataset_name=profile_dict.get('dataset_name'),
                                dataset_id=profile_dict.get('dataset_id'),
                                path=profile_dict.get('path'),
                                table_name=profile_dict.get('table_name'),
                                table_id=profile_dict.get('table_id'),
                                column_name=profile_dict.get('column_name'),
                                data_source=profile_dict.get('data_source'),
                                data_type=profile_dict.get('data_type'),
                                total_values=profile_dict.get('total_values_count'),
                                distinct_values_count=profile_dict.get('distinct_values_count'),
                                missing_values_count=profile_dict.get('missing_values_count'),
                                true_ratio=profile_dict.get('true_ratio'),
                                min_value=profile_dict.get('min_value'),
                                max_value=profile_dict.get('max_value'),
                                mean=profile_dict.get('mean'),
                                median=profile_dict.get('median'),
                                iqr=profile_dict.get('iqr'),
                                embedding=profile_dict.get('embedding'),
                                embedding_scaling_factor=profile_dict.get('embedding_scaling_factor'))
        return profile
    
    def is_numeric(self) -> bool:
        return self.get_data_type() in [ColumnDataType.INT.value, ColumnDataType.FLOAT.value]
    
    def is_textual(self) -> bool:
        return self.get_data_type() in [ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY.value,
                                        ColumnDataType.NATURAL_LANGUAGE_TEXT.value, ColumnDataType.STRING.value]
    
    def is_boolean(self) -> bool:
        return self.get_data_type() == ColumnDataType.BOOLEAN.value
    
    def is_float(self) -> bool:
        return self.get_data_type() == ColumnDataType.FLOAT.value
    
    def is_int(self) -> bool:
        return self.get_data_type() == ColumnDataType.INT.value
    
    def is_date(self) -> bool:
        return self.get_data_type() == ColumnDataType.DATE.value

    def is_named_entity(self) -> bool:
        return self.get_data_type() == ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY.value
    
    def is_natural_language_text(self) -> bool:
        return self.get_data_type() == ColumnDataType.NATURAL_LANGUAGE_TEXT.value
    
    def is_string(self) -> bool:
        return self.get_data_type() == ColumnDataType.STRING.value
    
    def get_column_id(self) -> str:
        return self.column_id

    def get_dataset_name(self) -> str:
        return self.dataset_name
        
    def get_dataset_id(self) -> str:
        return self.dataset_id

    def get_path(self) -> str:
        return self.path

    def get_table_name(self) -> str:
        return self.table_name
    
    def get_table_id(self) -> str:
        return self.table_id

    def get_column_name(self) -> str:
        return self.column_name
        
    def get_data_source(self) -> str:
        return self.data_source

    def get_data_type(self) -> str:
        return self.data_type

    def get_total_values_count(self) -> float:
        return self.total_values

    def get_distinct_values_count(self) -> float:
        return self.distinct_values_count

    def get_missing_values_count(self) -> float:
        return self.missing_values_count

    def get_true_ratio(self) -> float:
        return self.true_ratio

    def get_min_value(self) -> float:
        return self.min_value

    def get_max_value(self) -> float:
        return self.max_value

    def get_mean(self) -> float:
        return self.mean

    def get_median(self) -> float:
        return self.median

    def get_iqr(self) -> float:
        return self.iqr

    def get_embedding(self) -> list:
        return self.embedding

    def get_embedding_scaling_factor(self) -> float:
        return self.embedding_scaling_factor
    
    def set_column_id(self, column_id: float):
        self.column_id = column_id

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_path(self, path: str):
        self.path = path

    def set_table_name(self, table_name: str):
        self.table_name = table_name

    def set_column_name(self, column_name: str):
        self.column_name = column_name

    def set_data_type(self, data_type: str):
        self.data_type = data_type

    def set_total_values(self, total_values: float):
        self.total_values = total_values

    def set_distinct_values_count(self, unique_values: float):
        self.distinct_values_count = unique_values

    def set_missing_values_count(self, missing_values_count: float):
        self.missing_values_count = missing_values_count
    
    def set_true_ratio(self, true_ratio: float):
        self.true_ratio = true_ratio
        
    def set_min_value(self, min_value: float):
        self.min_value = min_value

    def set_max_value(self, max_value: float):
        self.max_value = max_value

    def set_mean(self, mean: float):
        self.mean = mean

    def set_median(self, median: float):
        self.median = median

    def set_iqr(self, iqr: float):
        self.iqr = iqr

    def set_embedding(self, embedding: list):
        self.embedding = embedding
        
    def set_embedding_scaling_factor(self, embedding_scaling_factor: float):
        self.embedding_scaling_factor = embedding_scaling_factor

    def __str__(self):
        return self.column_id + ': ' + str(self.embedding)
