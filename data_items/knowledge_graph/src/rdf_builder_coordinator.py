import os
import sys
import time
import json
import warnings

warnings.filterwarnings('ignore')
from rdf_builder import RDFBuilder
from elasticsearch import Elasticsearch, helpers, ElasticsearchException
from storage.elasticsearch_client import ElasticsearchClient



def load_json_to_elasticsearch(json_root='../../profiler/src/storage/meta_data/profiles/'):
    def load_json(json_root):
        for filename in os.listdir(json_root):
            if filename.endswith('.json'):
                with open(json_root + filename, 'r') as open_file:
                    yield json.load(open_file)

    if not os.path.exists(json_root):
        sys.exit("path: '{}' not found".format(json_root))
    else:
        column_count = len(os.listdir(json_root))
        status = '{} column profiles found'.format(column_count)
        print(status)
        es = Elasticsearch()
        status = 'attempting to store {} profiles on elasticsearch'.format(column_count)
        print(status)
        t = time.time()
        try:
            helpers.bulk(es, load_json(json_root))
        except ElasticsearchException as es_exception:
            print('Could not store the profiles because:\n' + str(es_exception))
        status = 'profiles stored successfully!\ttime taken: {}'.format(time.time() - t)
        print(status)


def refresh(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


def main(output_path=None):
    refresh('out/kglids_data_graph.ttls')
    load_json_to_elasticsearch()
    start_all = time.time()
    rdfBuilder = RDFBuilder()
    store = ElasticsearchClient()

    # Get all fields from store
    fields_gen = store.get_profile_attributes()

    # Network skeleton and hierarchical relations (table - field), etc
    print("\n• 1. Initializing nodes")
    start_schema = time.time()
    rdfBuilder.initialize_nodes(fields_gen)
    end_schema = time.time()
    print("• Done.\tTime taken: " + str(end_schema - start_schema))

    # semantic similarity (for all data types)
    print("\n• 2. Computing semantic similarity")
    start_schema_sim = time.time()
    schema_sim_index = rdfBuilder.build_semantic_sim_relation()
    end_schema_sim = time.time()
    print("• Done.\tTime taken: " + str(end_schema_sim - start_schema_sim))

    # content similarity for "T" (string relations) relation (minhash-based)
    print("\n• 3. Computing content similarity")
    start_text_sig_sim = time.time()
    st = time.time()
    string_types = ["T", "T_code", "T_email", "T_date", "T_org", "T_loc", "T_person"]

    for string_type in string_types:
        mh_signatures = store.get_profiles_minhash(string_type)
        et = time.time()
        # print("Time to extract minhash signatures from store: {0}".format(str(et - st)))
        # print("!!3 ({}) ".format(string_type) + str(et - st))
        print("\t Evaluating content similarity for: ", string_type)
        content_sim_index = rdfBuilder.build_content_sim_mh_text(mh_signatures)

    # content similarity for 'N' numerical relation
    print("\t Evaluating content similarity for: N")
    id_sig = store.get_num_stats()
    # networkbuilder.build_content_sim_relation_num(network, id_sig)
    rdfBuilder.build_content_sim_relation_num_overlap_distr(id_sig)
    # networkbuilder.build_content_sim_relation_num_overlap_distr_indexed(network, id_sig)
    end_num_sig_sim = time.time()
    print("• Done.\tTime taken: " + str(end_num_sig_sim - start_text_sig_sim))

    # content similarity (deep embeddings) for 'N' numerical relation
    print("\n• 4. Computing Deep embeddings")
    start_num_sig_sim_de = time.time()
    de_signatures = store.get_profiles_deep_embeddings()
    rdfBuilder.build_content_sim_de_num(de_signatures)
    end_num_sig_sim_de = time.time()
    print("• Done.\tTime taken: " + str(end_num_sig_sim_de - start_num_sig_sim_de))

    # Primary Key / Foreign key relation
    print("\n• 5. Computing PKFK relationships")
    start_pkfk = time.time()
    rdfBuilder.build_pkfk_relation()
    end_pkfk = time.time()
    print("• Done.\tTime taken: " + str(end_pkfk - start_pkfk))

    end_all = time.time()
    print("\nTotal time to build graph: " + str(end_all - start_all))


if __name__ == "__main__":
    # path = None
    path = 'out'
    '''
    if len(sys.argv) == 3:
        path = sys.argv[2]

    else:
        print("USAGE: ")
        print("python rdf_builder_coordinator.py --opath <path>")
        print("where opath must be writable by the process")
        exit()
    '''
    main(path)
