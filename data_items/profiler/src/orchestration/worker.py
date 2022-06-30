import itertools
import queue
import threading
from xmlrpc.client import boolean

from analysis.interpreter.interpreter import Interpreter
from analysis.profile_creator.profile_creator import ProfileCreator


class Worker(threading.Thread):

    def __init__(self, name: str, tables: queue, screenLock: threading, profile_save_base_dir):
        threading.Thread.__init__(self)
        self.name = name
        self.tables = tables
        self.screenLock = screenLock
        self.profile_save_base_dir = profile_save_base_dir


    def run(self):
        while True:
            table = self.tables.get()
            interpreter = Interpreter(table)
            profile_creator = ProfileCreator(table)

            # Interpret the tables
            textual_columns = interpreter.get_textual_columns()
            numerical_columns = interpreter.get_numerical_columns()
            boolean_columns = interpreter.get_boolean_columns()

            # Create profiles
            numerical_profiles = profile_creator.create_numerical_profiles(numerical_columns)
            textual_profiles = profile_creator.create_textual_profiles(textual_columns)
            boolean_profiles = [] # profile_creator.create_boolean_profiles(boolean_columns) # TODO: [Refactor] remove

            self.screenLock.acquire()
            print(self.name + " finished profiling " + table.get_table_name())
            self.screenLock.release()

            # store profiles on disk
            for profile in itertools.chain(numerical_profiles, textual_profiles, boolean_profiles):
                profile.save_profile(self.profile_save_base_dir)

            self.tables.task_done()
            self.screenLock.acquire()

            print(self.name + " Remaining tables " + str(self.tables.qsize()))
            self.screenLock.release()
            