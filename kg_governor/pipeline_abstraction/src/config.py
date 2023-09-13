import os


class AbstractionConfig:

    data_source_path = os.path.expanduser('~/projects/kglids/storage/data_sources/all_datasets/')
    output_graphs_path = os.path.expanduser('~/projects/kglids/storage/knowledge_graph/pipeline_abstraction/all_datasets/')


abstraction_config = AbstractionConfig()