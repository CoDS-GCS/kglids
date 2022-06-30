from data.tables.i_table import ITable


class CSVTable(ITable):

    def __init__(self, datasource: str, table_path: str, dataset_name: str, dataset_path: str, origin: str):
        self.datasource = datasource
        self.table_name = table_path.split('/')[-1]
        self.dataset_name = dataset_name
        self.table_path = table_path
        self.origin = origin

    def get_table_path(self):
        return self.table_path

    def get_dataset_name(self):
        return self.dataset_name

    def get_table_name(self):
        return self.table_name

    def get_origin(self):
        return self.origin
    
    def get_datasource(self):
        return self.datasource
