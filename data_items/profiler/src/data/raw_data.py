class RawData:
    def __init__(self, rid: float, origin: str, dataset_name: str, path: str, table_name: str, column_name: str,
                 values: list):
        self.rid = rid
        self.origin = origin
        self.dataset_name = dataset_name
        self.path = path
        self.table_name = table_name
        self.column_name = column_name
        self.values = values

    def get_rid(self) -> float:
        return self.rid

    def get_origin(self) -> float:
        return self.origin

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def get_path(self) -> str:
        return self.path

    def get_table_name(self) -> str:
        return self.table_name

    def get_column_name(self) -> str:
        return self.column_name

    def get_values(self) -> int:
        return self.values

    def set_rid(self, rid: float):
        self.rid = rid

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

    def set_values(self, values: list):
        self.values = values
