import queue
import threading

from data.utils.yaml_parser import YamlParser
from orchestration.utils import create_sources_from_datasets
from orchestration.worker import Worker


class Orchestrator:

    def __init__(self):
        self.tables = queue.Queue()  # will hold the different tables to be processed

    def create_tables(self, path: str):
        parser = YamlParser(path)
        parser.process_config_file()
        datasets = parser.get_datasets_info()
        datasource = parser.get_datasource_info()
        create_sources_from_datasets(datasource, datasets, self.tables)

    def process_tables(self, num_threads: int):
        screenLock = threading.Lock()
        for i in range(num_threads):
            # TODO: [Refactoring] save path should be taken from project config.
            worker = Worker(name=f'Thread {i}', tables=self.tables, screenLock=screenLock, 
                            profile_save_base_dir='storage/meta_data/profiles/')
            worker.setDaemon(True)
            worker.start()
        self.tables.join()

    def get_remaining_tables(self):
        return self.tables.qsize()
