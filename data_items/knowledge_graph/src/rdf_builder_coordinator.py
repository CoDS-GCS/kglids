import os
import sys
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')
from knowledge_graph_builder import KnowledgeGraphBuilder
from elasticsearch import Elasticsearch, helpers, ElasticsearchException
from storage.elasticsearch_client import ElasticsearchClient

# TODO: [Refactor] have main.py instead of builder coordinator
# TODO: [Refactor] combine with pipeline abstraction KG builder




def main():
    # TODO: [Refactor] read column profiles path from project config.py
    # TODO: [Refactor] add graph output path to project config.py
    
    start_all = datetime.now()
    knowledge_graph_builder = KnowledgeGraphBuilder(column_profiles_path='/home/mossad/projects/kglids/data_items/profiler/src/storage/metadata/profiles',
                                                    out_graph_path='out/kglids_data_items_graph.ttls')


    # Membership (e.g. table -> dataset) and metadata (e.g. min, max) triples
    print("\n• 1. Building Membership and Metadata triples")
    start_schema = datetime.now()
    knowledge_graph_builder.build_membership_and_metadata_subgraph()
    end_schema = datetime.now()
    print("• Done.\tTime taken: " + str(end_schema - start_schema))

    print("\n• 2. Computing column-column similarities")
    start_schema_sim = datetime.now()
    knowledge_graph_builder.generate_similarity_triples()
    end_schema_sim = datetime.now()
    print("• Done.\tTime taken: " + str(end_schema_sim - start_schema_sim))
    
    
    end_all = datetime.now()
    print("\nTotal time to build graph: " + str(end_all - start_all))


if __name__ == "__main__":
    main()
