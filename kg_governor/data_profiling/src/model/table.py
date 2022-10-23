class Table:

    def __init__(self, data_source: str, table_path: str, dataset_name: str):
        self.data_source = data_source
        self.table_path = table_path
        self.table_name = table_path.split('/')[-1]
        self.dataset_name = dataset_name

    def get_table_path(self):
        return self.table_path

    def get_dataset_name(self):
        return self.dataset_name

    def get_table_name(self):
        return self.table_name

    def get_data_source(self):
        return self.data_source

