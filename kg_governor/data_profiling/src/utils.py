import urllib

def encode(entity):
            return urllib.parse.quote_plus(entity)

def generate_id(datasource: str, dataset_name: str, table_name: str, column_name: str):
    return encode(datasource) + '/' + encode(dataset_name) + '/dataResource/' + encode(table_name) + '/' + encode(column_name)

def generate_table_id(datasource: str, dataset_name: str, table_name: str):
    return encode(datasource) + '/' + encode(dataset_name) + '/dataResource/' + encode(table_name) 

def generate_dataset_id(datasource: str, dataset_name: str):
    return encode(datasource) + '/' + encode(dataset_name)