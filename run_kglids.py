from kg_governor.data_profiling.profile_data import profile_data
from kg_governor.pipeline_abstraction.abstract_pipelines import abstract_pipelines
from kglids_config import KGLiDSConfig


def main():
    # run profiler
    # profile_data()

    # TODO: load embeddings into pgvector

    # run pipeline abstraction
    abstract_pipelines()

# create graphdb repo
# create postgres db
# run kg construction
# load graph into graphdb

if __name__ == '__main__':
    main()
