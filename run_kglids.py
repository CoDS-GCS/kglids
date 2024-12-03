from kglids_config import KGLiDSConfig
from kg_governor.data_profiling.profile_data import profile_data
from kg_governor.pipeline_abstraction.abstract_pipelines import abstract_pipelines
from kg_governor.data_global_schema_builder.build_data_global_schema import build_data_global_schema


def main():
    # run profiler
    profile_data()


    # # run pipeline abstraction
    abstract_pipelines()

    # run kg construction
    build_data_global_schema()



# create graphdb repo
# create postgres db
# load graph into graphdb
# load embeddings into pgvector

if __name__ == '__main__':
    main()
