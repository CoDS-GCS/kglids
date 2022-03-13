import os
import glob
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from orchestration.orchestrator import Orchestrator


def start_profiling():
    start_time = datetime.now()
    # TODO: [Implement] make config.py file containing configuration for the whole project
    # TODO: [Refactor] use config file for this path
    
    if os.path.exists('storage/metadata/profiles'):
        print('Deleting existing column profiles')
        shutil.rmtree('storage/metadata/profiles')
    
    os.makedirs('storage/metadata/profiles', exist_ok=True)
    
    orchestrator = Orchestrator()
    print('Creating tables')
    # TODO: [Refactor] use a more robust path to the config and add this configuration to global config.py
    orchestrator.create_tables('../config/config.yml')
    print('Processing tables')
    orchestrator.process_tables(num_threads=10)
    print('\n{} columns profiled!'.format(len(glob.glob('storage/metadata/profiles/**/*.json'))))
    print('Time to profile: ', datetime.now() - start_time)


start_profiling()
