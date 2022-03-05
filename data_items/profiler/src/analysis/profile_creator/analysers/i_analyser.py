from abc import ABC, abstractmethod


class IAnalyser(ABC):

    @abstractmethod
    def analyse_columns(self):
        pass

    @abstractmethod
    def get_profiles_info(self):
        pass
