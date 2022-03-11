import sys

sys.path.insert(0, '../../../src')

from data.utils.yaml_parser import YamlParser


def test_yaml_parser():
    yamlParser = YamlParser('config.yml')
    yamlParser.process_config_file()
    datasets = yamlParser.get_datasets_info()
    assert (len(datasets) == 2)
    dataset1 = datasets[0]
    dataset2 = datasets[1]

    # Check the first dataset
    assert (dataset1['name'] == 'dataset1')
    assert (dataset1['type'] == 'csv')
    assert (dataset1['path'] == '/path1/ds1/')
    assert (dataset1['origin'] == 'test')

    # Check the second dataset
    assert (dataset2['name'] == 'dataset2')
    assert (dataset2['type'] == 'csv')
    assert (dataset2['path'] == '/path2/ds2/')
    assert (dataset2['origin'] == 'test')
