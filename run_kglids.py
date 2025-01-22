from kglids_config import KGLiDSConfig
from kg_governor.data_profiling.profile_data import profile_data
from kg_governor.pipeline_abstraction.abstract_pipelines import abstract_pipelines
from kg_governor.data_global_schema_builder.build_data_global_schema import build_data_global_schema
from storage_utils.graphdb_utils import create_graphdb_repo, populate_pipeline_graphs, populate_data_global_schema_graph
from storage_utils.embedding_store_utils import create_embedding_db, populate_column_embeddings
def main():
    # run profiler
    profile_data()

    # run pipeline abstraction
    abstract_pipelines()

    # run kg construction
    build_data_global_schema()

    # create graphdb repo
    create_graphdb_repo(KGLiDSConfig.graphdb_endpoint, KGLiDSConfig.graphdb_repo_name)

    # load graph into graphdb
    populate_data_global_schema_graph(KGLiDSConfig.data_global_schema_graph_out_path,
                                      KGLiDSConfig.graphdb_repo_name,
                                      KGLiDSConfig.graphdb_endpoint,
                                      KGLiDSConfig.graphdb_import_path)
    populate_pipeline_graphs(KGLiDSConfig.pipeline_graphs_out_path,
                             KGLiDSConfig.graphdb_endpoint,
                             KGLiDSConfig.graphdb_repo_name)

    # create postgres db
    create_embedding_db(KGLiDSConfig.column_embeddings_db_name)

    # load embeddings into pgvector
    populate_column_embeddings(KGLiDSConfig.profiles_out_path, KGLiDSConfig.column_embeddings_db_name)


if __name__ == '__main__':
    main()
