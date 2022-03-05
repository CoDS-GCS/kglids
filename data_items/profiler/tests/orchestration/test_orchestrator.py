import sys

sys.path.insert(0, '../../src')

from orchestration.orchestrator import Orchestrator


def test_process_tables():
    orchestrator = Orchestrator()
    orchestrator.create_tables('../resources/config.yml')
    num_remaining_tables = orchestrator.get_remaining_tables()
    assert (num_remaining_tables == 2)
    orchestrator.process_tables(3)
    num_remaining_tables = orchestrator.get_remaining_tables()

    assert (num_remaining_tables == 0)
