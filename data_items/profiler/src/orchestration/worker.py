import queue
import threading
from xmlrpc.client import boolean

from analysis.interpreter.interpreter import Interpreter
from analysis.profile_creator.profile_creator import ProfileCreator
# from storage.i_documentDB import IDocumentDB
from storage.i_documentDB_disk import IDocumentDB_disk

class Worker(threading.Thread):

    def __init__(self, name: str, tables: queue, screenLock: threading, document_db_disk: IDocumentDB_disk):
        threading.Thread.__init__(self)
        self.name = name
        self.tables = tables
        self.screenLock = screenLock
        #self.document_db = document_db
        self.document_db_disk = document_db_disk

    def run(self):
        while True:
            table = self.tables.get()
            interpreter = Interpreter(table)
            profile_creator = ProfileCreator(table)

            self.screenLock.acquire()
            print(self.name + " is interpreting " + table.get_table_name())
            self.screenLock.release()
            # Interpret the tables
            textual_columns = interpreter.get_textual_columns()
            numerical_columns = interpreter.get_numerical_columns()
            boolean_columns = interpreter.get_boolean_columns()
            self.screenLock.acquire()
            print(self.name + " finished interpreting " + table.get_table_name())
            self.screenLock.release()
            # Create profiles

            self.screenLock.acquire()
            print(self.name + " is profiling " + table.get_table_name())
            self.screenLock.release()
            numerical_profiles_iterator = profile_creator.create_numerical_profiles(numerical_columns)
            textual_profiles_iterator = profile_creator.create_textual_profiles(textual_columns)
            booleab_profiles_iterator = profile_creator.create_boolean_profiles(boolean_columns)

            self.screenLock.acquire()
            print(self.name + " finished profiling " + table.get_table_name())
            self.screenLock.release()

            # store profiles on disk
            self.document_db_disk.store_profiles_disk(numerical_profiles_iterator)
            self.document_db_disk.store_profiles_disk(textual_profiles_iterator)
            self.document_db_disk.store_profiles_disk(booleab_profiles_iterator)
            # store raw data on disk
            #raw_data_iterator = interpreter.get_raw_data()
            #self.document_db_disk.store_data_disk(raw_data_iterator)

            self.tables.task_done()
            self.screenLock.acquire()

            print(self.name + " Remaining tables " + str(self.tables.qsize()))
            self.screenLock.release()
