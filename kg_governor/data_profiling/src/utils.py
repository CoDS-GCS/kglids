import urllib


def encode(entity):
    return urllib.parse.quote_plus(entity)


def generate_column_id(data_source: str, dataset_name: str, table_name: str, column_name: str):
    return encode(data_source) + '/' + encode(dataset_name) + encode(table_name) + '/' + encode(column_name)


def generate_table_id(data_source: str, dataset_name: str, table_name: str):
    return encode(data_source) + '/' + encode(dataset_name) + encode(table_name) 


def generate_dataset_id(data_source: str, dataset_name: str):
    return encode(data_source) + '/' + encode(dataset_name)
