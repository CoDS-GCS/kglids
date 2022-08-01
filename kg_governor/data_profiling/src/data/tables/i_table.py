from abc import ABC, abstractmethod


class ITable(ABC):

    @abstractmethod
    def get_table_path(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    @abstractmethod
    def get_table_name(self):
        pass

    @abstractmethod
    def get_origin(self):
        pass

    @abstractmethod
    def get_datasource(self):
        pass

