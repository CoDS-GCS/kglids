import os
import time
import uuid
from collections import defaultdict
from math import isinf
from multiprocessing.pool import ThreadPool
import shutil
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import DBSCAN
from enums.relation import Relation
from rdf_resource import RDFResource
from triplet import Triplet
from utils import generate_label
from word_embedding.word_embeddings_services import WordEmbeddingServices

CHUNK_SIZE = 500000


def _generate_id(name, dic):
    if name in dic.keys():
        return dic[name]
    else:
        return str(uuid.uuid1().int)


class RDFBuilder:
    global olids
    olids = 'olids'

    def __init__(self):
        self.word_embeddings = WordEmbeddingServices()
        self.__triplets = []
        self.__docs = []
        self.__namespaces = {'olids': 'http://kglids.org/',
                             'schema': 'http://schema.org/',
                             'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                             'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                             'owl': 'http://www.w3.org/2002/07/owl#',
                             'dct': 'http://purl.org/dc/terms/'}
        self._column_id_to_name = dict()
        self._table_id_to_name = dict()
        self._source_id_to_name = dict()
        self._dataset_id_to_name = dict()
        self.__source_ids = defaultdict(list)
        self.__neighbors = defaultdict(dict)
        self.__id_to_labels = dict()

    def initialize_nodes(self, fields):

        def create_triplets(rdf_source, predicate, objct):
            triplet = Triplet(rdf_source,
                              predicate,
                              RDFResource(objct))
            self.__triplets.append(triplet)

        def create_predicate(relation, namespace):
            return RDFResource(relation, False, self.__namespaces[namespace])

        t = time.time()

        n_cols = 0
        for (
        column_id, origin, dataset_name, dataset_id, table_name, table_id, datasource, column_name, total_values_count,
        distinct_values_count,
        missing_values_count, data_type, median, minValue, maxValue, path) in fields:

            self._column_id_to_name[column_id] = (dataset_name, table_name, column_name, data_type)
            self.__source_ids[table_name].append(column_id)
            self.__docs.append(column_name)

            column_node = RDFResource(column_id, False, self.__namespaces[olids])
            table_node = RDFResource(table_id, False, self.__namespaces[olids])
            dataset_node = RDFResource(dataset_id, False, self.__namespaces[olids])
            source_node = RDFResource(datasource, False, self.__namespaces[olids])
            col_label = generate_label(column_name, 'en')
            self.__id_to_labels[column_id] = [col_label.get_text(), data_type, table_name]
            '''
            n_cols = n_cols + 1
            print("Processed columns: ", n_cols)
            print("dict: ", len(self.__id_to_labels.keys()))
            '''
            create_triplets(column_node, create_predicate('type', 'schema'), data_type)
            create_triplets(column_node, create_predicate('name', 'schema'), column_name)
            create_triplets(column_node, create_predicate('totalVCount', 'schema'), total_values_count)
            create_triplets(column_node, create_predicate('distinctVCount', 'schema'), distinct_values_count)
            create_triplets(column_node, create_predicate('missingVCount', 'schema'), missing_values_count)
            create_triplets(column_node, create_predicate('label', 'rdfs'), col_label)
            if data_type == 'N':
                create_triplets(column_node, create_predicate('median', 'schema'), median)
                create_triplets(column_node, create_predicate('maxValue', 'schema'), maxValue)
                create_triplets(column_node, create_predicate('minValue', 'schema'), minValue)

            create_triplets(column_node, create_predicate(Relation.isPartOf.name, 'dct'), table_node)
            create_triplets(column_node, create_predicate('type', 'rdf'),
                            RDFResource('data/column', False, self.__namespaces[olids]))

            if not (table_name in self._table_id_to_name.keys() and dataset_name in self._dataset_id_to_name.keys()):
                table_label = generate_label(table_name, 'en')
                create_triplets(table_node, create_predicate('name', 'schema'), table_name)
                create_triplets(table_node, create_predicate('label', 'rdfs'), table_label)
                create_triplets(table_node, create_predicate('path', 'olids'), path)
                create_triplets(table_node, create_predicate(Relation.isPartOf.name, 'dct'), dataset_node)
                create_triplets(table_node, create_predicate('type', 'rdf'),
                                RDFResource('data/table', False, self.__namespaces[olids]))
                if dataset_name not in self._dataset_id_to_name.keys():
                    dataset_label = generate_label(dataset_name, 'en')
                    create_triplets(dataset_node, create_predicate('name', 'schema'), dataset_name)
                    create_triplets(dataset_node, create_predicate('label', 'rdfs'), dataset_label)
                    create_triplets(dataset_node, create_predicate('type', 'rdf'),
                                    RDFResource('data/dataset', False, self.__namespaces[olids]))

                    if datasource not in self._source_id_to_name.keys():
                        source_label = generate_label(datasource, 'en')
                        create_triplets(source_node, create_predicate('name', 'schema'), datasource)
                        create_triplets(source_node, create_predicate('label', 'rdfs'), source_label)
                        create_triplets(source_node, create_predicate('type', 'rdf'),
                                        RDFResource('data/source', False, self.__namespaces[olids]))
                        self._source_id_to_name[datasource] = datasource
                    self._dataset_id_to_name[dataset_name] = dataset_id
                self._table_id_to_name[table_name] = table_id

            cardinality_ratio = None
            if float(total_values_count) > 0:
                cardinality_ratio = float(distinct_values_count) / float(total_values_count)
                create_triplets(column_node, create_predicate(Relation.cardinality.name, 'owl'), cardinality_ratio)
                self.__neighbors[column_id].update({'cardinality': cardinality_ratio})

            '''
            # append origins to tables
            for table_id in self._table_id_to_origin.keys():
                table_node = RDFResource(table_id, False, self.__namespaces[olids])
                create_triplets(table_node,
                                create_predicate('origin', olids),
                                self._table_id_to_origin[table_id])
            '''
            self.dump_and_release_memory()

        self.dump_into_file(self.__triplets)
        self.__triplets = []


    def neighbors_id(self, arg, relation):
        if isinstance(arg, str):
            nid = arg
        elif isinstance(arg, float):
            nid = arg
        elif isinstance(arg, int):
            nid = arg
        else:
            raise ValueError('arg must be either str, int, or float')
        nid = str(nid)
        data = []

        if 'deep_embeddings' not in str(relation):
            relation_name = relation.name
        elif 'deep_embeddings' in str(relation):
            relation_name = Relation.deep_embeddings.name + '/' + Relation.contentSimilarity.name

        if not relation_name in self.__neighbors[nid]:
            return data
        neighbours = self.__neighbors[nid][relation_name]
        '''
        if not relation.name in self.__neighbors[nid]:
            return data
        neighbours = self.__neighbors[nid][relation.name]
        '''
        for neighbor_id, score in neighbours:
            (db_name, source_name, field_name, data_type) = self._column_id_to_name[neighbor_id]
            data.append((neighbor_id, db_name, source_name, field_name, score))
        return data

    def get_triplets(self):
        return list(set([str(t) for t in self.__triplets]))

    def dump_and_release_memory(self):
        if len(self.__triplets) >= CHUNK_SIZE:
            self.dump_into_file(self.__triplets)

            self.__triplets = []

    def dump_into_file(self, triplets, filename='out/kglids_data_graph.ttls', mode='a'):
        # make sure the directory exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, mode, encoding='utf-8') as f:
            for item in triplets:
                f.write("%s\n" % item)
        self.__triplets = []

    def append_tmp_similarities_to_graph(self, tmp_dir, graph_filename='out/kglids_data_graph.ttls'):
        # make sure tmp_dir ends with /
        tmp_dir = tmp_dir.rstrip('/') + '/'
        tmp_files = [tmp_dir + i for i in os.listdir(tmp_dir) if i.endswith('ttls')]

        with open(graph_filename, 'a', encoding='utf-8') as fout:
            for tmp_file in tmp_files:
                # read the content of tmp similarity file
                with open(tmp_file, 'r', encoding='utf-8') as fin:
                    content = fin.read()
                # append it to graph file
                fout.write(content)
                # remove the tmp file
                os.remove(tmp_file)

    def set_neighbors(self, relation, nid1, nid2, score):
        if not relation in self.__neighbors[nid1]:
            self.__neighbors[nid1].update({relation: [(nid2, score)]})
        else:
            if not (nid2, score) in self.__neighbors[nid1][relation]:
                self.__neighbors[nid1][relation].append((nid2, score))

    def build_semantic_sim_relation(self):

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[olids])
            nestedPredicate = RDFResource(Relation.semanticSimilarity.name, False, self.__namespaces[olids])
            nestedObject = RDFResource(nid2, False, self.__namespaces[olids])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[olids])
            objct = RDFResource(score)

            self.set_neighbors(Relation.semanticSimilarity.name, nid1, nid2, score)
            self.set_neighbors(Relation.semanticSimilarity.name, nid2, nid1, score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            return [triplet1, triplet2]

        t = time.time()

        self.__triplets = []  # to remove

        status = "\t> total number of columns: " + str(len(self.__id_to_labels))
        print(status)
        wiki_model = self.word_embeddings.get_wiki_model()

        def get_ids_by_dtype(id_to_labels, dt):

            pairs = {}
            for key, value in id_to_labels.items():
                if value[1] == dt:
                    pairs[key] = value

            return pairs

        numerical_pairs = get_ids_by_dtype(self.__id_to_labels, 'N')
        boolean_pairs = get_ids_by_dtype(self.__id_to_labels, 'B')
        string_pairs = get_ids_by_dtype(self.__id_to_labels, 'T')
        date_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_date')
        email_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_email')
        code_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_code')
        loc_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_loc')
        org_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_org')
        person_pairs = get_ids_by_dtype(self.__id_to_labels, 'T_person')
        status = "\t• breakdown - numerical cols: " + str(len(numerical_pairs)) + " | boolean cols: " + str(
            len(boolean_pairs)) + " | string cols: " + str(len(string_pairs)) + " | code cols: " + str(
            len(code_pairs)) + " | email cols: " + str(len(email_pairs)) + " | date cols: " + str(
            len(date_pairs)) + "\n | organization cols: " + str(len(org_pairs)) + " | location cols: " + str(
            len(loc_pairs)) + " | person cols: " + str(len(person_pairs))
        print(status)

        total_comparisons = []

        def sim_relations(pairs: dict):
            if not len(pairs):
                return

            start = time.time()
            num_comparisons = 0.5 * len(pairs) * (len(pairs) + 1)
            total_comparisons.append(num_comparisons)
            columns_ids_and_labels = list(pairs.items())
            string_data_type = columns_ids_and_labels[0][1][1]

            def get_similarity_triplets_for_column(idx):
                c1_id, c1_label = columns_ids_and_labels[idx]
                similarity_triplets = []
                for j in range(idx + 1, len(columns_ids_and_labels)):
                    c2_id, c2_label = columns_ids_and_labels[j]

                    distance = wiki_model.get_distance_between_column_labels(c2_label[0], c1_label[0])
                    if distance >= 0.5:
                        similarity_triplets.extend(create_triplets(c1_id, c2_id, distance))

                if similarity_triplets:
                    # dump the similarities of the column to a temporary file
                    f_id = c1_id.replace(r'/', '_')
                    self.dump_into_file(similarity_triplets, f'tmp/semantic_similarities/col_{f_id}.ttls', 'w')

            # map the function to column ids/labels in parallel
            pool = ThreadPool(os.cpu_count())
            list(tqdm(pool.imap_unordered(get_similarity_triplets_for_column, range(len(columns_ids_and_labels))),
                      total=len(columns_ids_and_labels)))

            status = "\t~ summary for '{}' column's semantic similarity - time taken: [{}] | comparisons: {}".format(
                string_data_type, time.time() - start, num_comparisons)
            print(status)

        # recreate the tmp directory if it exists
        shutil.rmtree('tmp/semantic_similarities', ignore_errors=True)
        os.makedirs('tmp/semantic_similarities')

        status = "\t> evaluating numerical semantic similarity"
        print(status)

        sim_relations(numerical_pairs)

        status = "\t> evaluating boolean semantic similarity"
        print(status)

        sim_relations(boolean_pairs)

        status = "\t> evaluating string semantic similarity"
        print(status)
        sim_relations(string_pairs)

        status = "\t> evaluating date semantic similarity"
        print(status)
        sim_relations(date_pairs)

        status = "\t> evaluating code semantic similarity"
        print(status)
        sim_relations(code_pairs)

        status = "\t> evaluating email semantic similarity"
        print(status)
        sim_relations(email_pairs)

        status = "\t> evaluating organization semantic similarity"
        print(status)
        sim_relations(org_pairs)

        status = "\t> evaluating location semantic similarity"
        print(status)
        sim_relations(loc_pairs)

        status = "\t> evaluating person semantic similarity"
        print(status)
        sim_relations(person_pairs)

        # now the similarities are generated, append them to the graph from the tmp files
        self.append_tmp_similarities_to_graph('tmp/semantic_similarities/')

    def build_content_sim_mh_text(self, mh_signatures):

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[olids])
            nestedPredicate = RDFResource(Relation.contentSimilarity.name, False, self.__namespaces[olids])
            nestedObject = RDFResource(nid2, False, self.__namespaces[olids])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[olids])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.set_neighbors(Relation.contentSimilarity.name, nid1, nid2, score)
            self.set_neighbors(Relation.contentSimilarity.name, nid2, nid1, score)

            self.__triplets.extend([triplet1, triplet2])

        self.__triplets = []
        # Materialize signatures for convenience
        mh_sig_obj = []

        content_index = MinHashLSH(threshold=0.7, num_perm=512)
        print("\t fetching minhash values")
        # Create minhash objects and index
        for nid, mh_sig in tqdm(mh_signatures):
            mh_obj = MinHash(num_perm=512)
            # print(type(mh_sig), len(mh_sig))
            mh_array = np.asarray(mh_sig, dtype=int)
            mh_obj.hashvalues = mh_array
            content_index.insert(nid, mh_obj)
            mh_sig_obj.append((nid, mh_obj))

        # Query objects
        print("\t generating pairs")
        for nid, mh_obj in tqdm(mh_sig_obj):
            res = content_index.query(mh_obj)
            for r_nid in res:
                if r_nid != nid:
                    r_mh_obj = list(filter(lambda x: x[0] == r_nid, mh_sig_obj))[0][1]
                    distance = mh_obj.jaccard(r_mh_obj)
                    create_triplets(nid, r_nid, distance)

                self.dump_and_release_memory()

        self.dump_into_file(self.__triplets)
        self.__triplets = []
        return content_index

    def build_content_sim_relation_num_overlap_distr(self, id_sig):

        def compute_overlap(ref_left, ref_right, left, right):
            ov = 0
            if left >= ref_left and right <= ref_right:
                ov = float((right - left) / (ref_right - ref_left))
            elif left >= ref_left and left <= ref_right:
                domain_ov = ref_right - left
                ov = float(domain_ov / (ref_right - ref_left))
            elif right <= ref_right and right >= ref_left:
                domain_ov = right - ref_left
                ov = float(domain_ov / (ref_right - ref_left))
            return float(ov)

        def create_triplets(nid1, nid2, score, inddep=False):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[olids])
            nestedObject = RDFResource(nid2, False, self.__namespaces[olids])

            if inddep is False:
                nestedPredicate = RDFResource(Relation.contentSimilarity.name, False, self.__namespaces[olids])
                self.set_neighbors(Relation.contentSimilarity.name, nid1, nid2, score)
                self.set_neighbors(Relation.contentSimilarity.name, nid2, nid1, score)
            else:
                nestedPredicate = RDFResource(Relation.inclusionDependency.name, False, self.__namespaces[olids])
                self.set_neighbors(Relation.inclusionDependency.name, nid1, nid2, score)
                self.set_neighbors(Relation.inclusionDependency.name, nid2, nid1, score)

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[olids])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.__triplets.extend([triplet1, triplet2])

        def get_info_for(nids):
            info = []
            for nid in nids:
                db_name, source_name, field_name, data_type = self._column_id_to_name[nid]
                info.append((nid, db_name, source_name, field_name))
            return info

        self.__triplets = []
        overlap = 0.95

        fields = []
        domains = []
        stats = []
        for c_k, (c_median, c_iqr, c_min_v, c_max_v) in id_sig:
            fields.append(c_k)
            domain = (c_median + c_iqr) - (c_median - c_iqr)
            domains.append(domain)
            extreme_left = c_median - c_iqr
            min = c_min_v
            extreme_right = c_median + c_iqr
            max = c_max_v
            stats.append((min, extreme_left, extreme_right, max))

        zipped_and_sorted = sorted(zip(domains, fields, stats), reverse=True)
        candidate_entries = [(y, x, z[0], z[1], z[2], z[3]) for (x, y, z) in zipped_and_sorted]
        single_points = []
        for ref in candidate_entries:
            ref_nid, ref_domain, ref_x_min, ref_x_left, ref_x_right, ref_x_max = ref

            if ref_nid == '2314808454':
                debug = True

            if ref_domain == 0:
                single_points.append(ref)

            info1 = get_info_for([ref_nid])

            (nid, db_name, source_name, field_name) = info1[0]
            for entry in candidate_entries:
                candidate_nid, candidate_domain, candidate_x_min, candidate_x_left, candidate_x_right, candidate_x_max = entry
                if candidate_nid == '1504465753':
                    debug = True

                if candidate_nid == ref_nid:
                    continue

                if ref_domain == 0:
                    continue
                # Check for filtered inclusion dependencies first
                if isinstance(candidate_domain, float) or isinstance(candidate_domain, int):  # Filter these out
                    # Check ind. dep.
                    info2 = get_info_for([candidate_nid])
                    (_, _, sn1, fn1) = info1[0]
                    (_, _, sn2, fn2) = info2[0]
                    if isinf(float(ref_x_min)) or isinf(float(ref_x_max)) or isinf(float(candidate_x_max)) or isinf(
                            float(candidate_x_min)):
                        continue
                    if candidate_x_min >= ref_x_min and candidate_x_max <= ref_x_max:
                        # inclusion relation
                        actual_overlap = compute_overlap(ref_x_left, ref_x_right, candidate_x_left,
                                                         candidate_x_right)

                        if actual_overlap >= 0.90:
                            create_triplets(candidate_nid, ref_nid, actual_overlap, inddep=True)

                    self.dump_and_release_memory()
                # if float(candidate_domain / ref_domain) <= overlap:
                # There won't be a content sim relation -> not even the entire domain would overlap more than the th.
                #    break
                self.dump_into_file(self.__triplets)
                actual_overlap = compute_overlap(ref_x_left, ref_x_right, candidate_x_left, candidate_x_right)
                if actual_overlap >= overlap:
                    create_triplets(candidate_nid, ref_nid, actual_overlap)
        self.dump_into_file(self.__triplets)
        # Final clustering for single points
        fields = []
        medians = []

        for (nid, domain, x_min, x_left, x_right, x_max) in single_points:
            median = x_right - float(x_right / 2)
            fields.append(nid)
            medians.append(median)

        x_median = np.asarray(medians)
        x_median = x_median.reshape(-1, 1)

        # At this point, we may have not found any points at all, in which case we can
        # safely exit
        if len(x_median) == 0:
            self.dump_into_file(self.__triplets)
            # self.__triplets_to_dump.extend(self.__triplets)
            return

        db_median = DBSCAN(eps=0.1, min_samples=2).fit(x_median)
        labels_median = db_median.labels_
        n_clusters = len(set(labels_median)) - (1 if -1 in labels_median else 0)
        # print("#clusters: " + str(n_clusters))

        clusters_median = defaultdict(list)
        for i in range(len(labels_median)):
            clusters_median[labels_median[i]].append(i)

        for k, v in clusters_median.items():
            if k == -1:
                continue
            # print("Cluster: " + str(k))
            for el in v:
                nid = fields[el]
                info = get_info_for([nid])
                (nid, db_name, source_name, field_name) = info[0]
                # print(source_name + " - " + field_name + " median: " + str(medians[el]))
                for el2 in v:
                    if el != el2:
                        nid1 = fields[el]
                        nid2 = fields[el2]
                        create_triplets(nid1, nid2, overlap)
                self.dump_and_release_memory()

        self.dump_into_file(self.__triplets)

    def build_content_sim_de_num(self, de_signatures):

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[olids])
            nestedObject = RDFResource(nid2, False, self.__namespaces[olids])

            nestedPredicate = RDFResource(Relation.deep_embeddings.name + '/' + Relation.contentSimilarity.name, False,
                                          self.__namespaces[olids])
            self.set_neighbors(Relation.deep_embeddings.name + '/' + Relation.contentSimilarity.name, nid1, nid2, score)
            self.set_neighbors(Relation.deep_embeddings.name + '/' + Relation.contentSimilarity.name, nid2, nid1, score)

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[olids])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.__triplets.extend([triplet1, triplet2])

        threshold_deep_embeddings = 0.95
        total_matches = 0

        def cal_similarity_de(embedding1: list, embedding2: list):
            cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
            return cos_sim

        self.__triplets = []

        for nid1, de_sig in tqdm(de_signatures):
            for nid2, de_sig1 in de_signatures:

                if nid1 != nid2:
                    score = cal_similarity_de(de_sig, de_sig1)
                    if score >= threshold_deep_embeddings:
                        total_matches = total_matches + 1
                        create_triplets(nid1, nid2, score)

            self.dump_and_release_memory()

        self.dump_into_file(self.__triplets)

    def build_pkfk_relation(self):

        def get_data_type_of(nid):
            _, _, _, data_type = self._column_id_to_name[nid]
            return data_type

        def get_neighborhood(n):
            neighbors = []
            data_type = get_data_type_of(n)
            if data_type == "N":
                neighbors = self.neighbors_id(n, Relation.inclusionDependency)
                # neighbors = self.neighbors_id(n, Relation.deep_embeddings.name + '/' + Relation.contentSimilarity.name)
            if data_type == 'T' or data_type == 'T_date' or data_type == 'T_org' or data_type == 'T_code' or data_type == 'T_person' or data_type == 'T_loc' or data_type == 'T_email':
                neighbors = self.neighbors_id(n, Relation.contentSimilarity)

            return neighbors

        def iterate_ids():
            for k, _ in self._column_id_to_name.items():
                yield k

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[olids])
            nestedPredicate = RDFResource(Relation.pkfk.name, False, self.__namespaces[olids])
            nestedObject = RDFResource(nid2, False, self.__namespaces[olids])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[olids])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)


            self.__triplets.extend([triplet1, triplet2])

        self.__triplets = []
        for n in iterate_ids():
            if not 'cardinality' in self.__neighbors[n]:
                continue
            n_card = self.__neighbors[n]['cardinality']
            if n_card > 0.60:  # Early check if this is a candidate
                neighborhood = get_neighborhood(n)
                for ne in neighborhood:
                    if ne[0] is not n:
                        ne_card = self.__neighbors[ne[0]]['cardinality']
                        if n_card > ne_card:
                            highest_card = n_card
                        else:
                            highest_card = ne_card
                        # if ne_card < 0.5:
                        create_triplets(n, ne[0], highest_card)
                        # print(str(n) + " -> " + str(ne))
                    self.dump_and_release_memory()
                self.dump_into_file(self.__triplets)
        self.dump_into_file(self.__triplets)
