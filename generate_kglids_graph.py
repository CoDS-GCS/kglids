from kg_governor.data_profiling.profile_data import profile_data
from kglids_config import KGLiDSConfig


def main():
    profile_data(data_source=KGLiDSConfig.data_source, data_source_path=KGLiDSConfig.data_source_path,
                 profiles_out_path=KGLiDSConfig.profiles_out_path,
                 replace_existing_profiles=KGLiDSConfig.replace_existing_profiles,
                 is_spark_local_mode=KGLiDSConfig.is_spark_local_mode, spark_n_workers=KGLiDSConfig.spark_n_workers,
                 spark_max_memory=KGLiDSConfig.spark_max_memory)


# create graphdb repo
# create postgres db
# run pipeline abstraction
# run profiler
# run kg construction
# load graph into graphdb
# load embeddings into pgvector

if __name__ == '__main__':
    main()
