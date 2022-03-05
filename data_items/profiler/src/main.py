import time
import glob
import os
import warnings
warnings.filterwarnings('ignore')
from orchestration.orchestrator import Orchestrator


def start_profiling():
    start_time = time.time()
    if not os.path.exists('storage/meta_data/profiles'):
        os.makedirs('storage/meta_data/profiles')
    elif os.listdir('storage/meta_data/profiles/') != 0:
        files = glob.glob('storage/meta_data/profiles/*.json')
        print('Refreshing meta data!')
        for f in files:
            os.remove(f)

    orchestrator = Orchestrator()
    print('Creating tables')
    orchestrator.create_tables('../config/config.yml')
    print('Processing tables')
    orchestrator.process_tables(10)
    print('\n{} columns profiled!'.format(len(glob.glob('storage/meta_data/profiles/*.json'))))
    print('Time to profile: ', time.time() - start_time)


start_profiling()
