import os
import random
import json
import string

class ColumnProfile:
    # TODO: [Refactor] remove unneeded stats
    # TODO: [Refactor] combine minhash and deep_embedding attribute to embedding (minhash for strings and DDE for numerical)    
    def __init__(self, column_id: float, origin: str, dataset_name: str, dataset_id: str, path: str, table_name: str,
                 table_id: str, column_name: str, datasource: str, data_type: str,
                 total_values: float, distinct_values_count: float, missing_values_count: float, min_value: float,
                 max_value: float, mean: float, median: float, iqr: float,
                 minhash: list, deep_embeddings: list):
        self.column_id = column_id
        self.origin = origin
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.path = path
        self.table_name = table_name
        self.table_id = table_id
        self.datasource = datasource
        self.column_name = column_name
        self.data_type = data_type
        self.total_values = total_values
        self.distinct_values_count = distinct_values_count
        self.missing_values_count = missing_values_count
        self.max_value = max_value
        self.min_value = min_value
        self.mean = mean
        self.median = median
        self.iqr = iqr
        self.minhash = minhash
        self.deep_embeddings = deep_embeddings

    def to_dict(self):
        # TODO: [Refactor] rename these keys
        profile_dict = {'column_id': self.get_column_id(), 
                        'origin': self.get_origin(), 
                        'datasetName': self.get_dataset_name(),
                        'datasetid': self.get_dataset_id(), 
                        'path': self.get_path(),
                        'tableName': self.get_table_name(), 
                        'tableid': self.get_table_id(), 
                        'columnName': self.get_column_name(),
                        'datasource': self.get_datasource(),
                        'dataType': self.get_data_type(), 
                        'totalValuesCount': self.get_total_values_count(),
                        'distinctValuesCount': self.get_distinct_values_count(),
                        'missingValuesCount': self.get_missing_values_count(),
                        'minValue': self.get_min_value(), 
                        'maxValue': self.get_max_value(), 
                        'avgValue': self.get_mean(),
                        'median': self.get_median(), 
                        'iqr': self.get_iqr(),
                        'minhash': self.get_minhash(),
                        'deep_embeddings': self.get_deep_embeddings()}
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
                                origin=profile_dict.get('origin'),
                                dataset_name=profile_dict.get('datasetName'),
                                dataset_id=profile_dict.get('datasetid'),
                                path=profile_dict.get('path'),
                                table_name=profile_dict.get('tableName'),
                                table_id=profile_dict.get('tableid'),
                                column_name=profile_dict.get('columnName'),
                                datasource=profile_dict.get('datasource'),
                                data_type=profile_dict.get('dataType'),
                                total_values=profile_dict.get('totalValuesCount'),
                                distinct_values_count=profile_dict.get('distinctValuesCount'),
                                missing_values_count=profile_dict.get('missingValuesCount'),
                                min_value=profile_dict.get('minValue'),
                                max_value=profile_dict.get('maxValue'),
                                mean=profile_dict.get('avgValue'),
                                median=profile_dict.get('median'),
                                iqr=profile_dict.get('iqr'),
                                minhash=profile_dict.get('minhash'),
                                deep_embeddings=profile_dict.get('deep_embeddings'))
        return profile
    
    def get_column_id(self) -> float:
        return self.column_id

    def get_origin(self) -> str:
        return self.origin

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
        
    def get_datasource(self) -> str:
        return self.datasource

    def get_data_type(self) -> str:
        return self.data_type

    def get_total_values_count(self) -> float:
        return self.total_values

    def get_distinct_values_count(self) -> float:
        return self.distinct_values_count

    def get_missing_values_count(self) -> float:
        return self.missing_values_count

    def get_minhash(self) -> list:
        return self.minhash

    def get_deep_embeddings(self) -> list:
        return self.deep_embeddings

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

    def set_column_id(self, column_id: float):
        self.column_id = column_id

    def set_origin(self, origin: str):
        self.origin = origin

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

    def set_minhash(self, minhash: list):
        self.minhash = minhash

    def set_deep_embeddings(self, deep_embeddings: list):
        self.deep_embeddings = deep_embeddings

    def __str__(self):
        return self.table_name + ': ' + str(self.minhash) if self.minhash else str(self.deep_embeddings)
