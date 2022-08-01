import logging
import sys


class Logger:

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def create_console_logger(self):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        return self.logger
