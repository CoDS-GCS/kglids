from abc import ABC, abstractmethod


class IDocumentDB_disk(ABC):

    # @abstractmethod
    # def close_db(self):
    #     pass

    @abstractmethod
    # def store_data_disk(self, rawData: list):
    #     pass

    @abstractmethod
    def store_profiles_disk(self, profiles: list):
        pass

    # @abstractmethod
    # def delete_index(self, index: str):
    #     pass
