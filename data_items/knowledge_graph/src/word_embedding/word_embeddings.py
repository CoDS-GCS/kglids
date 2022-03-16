import numpy as np
import statistics
import itertools

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import NotFoundError


class WordEmbeddings:
    def __init__(self):
        self.es = Elasticsearch()

    def load_vocab_to_elasticsearch(self, path_to_raw_embeddings):
        with open(path_to_raw_embeddings, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        elasticsearch_payload = []
        for line in lines:
            split_line = line.strip().split()
            word = split_line[0]
            embedding = np.array([float(i) for i in split_line[1:]])
            # normalize to unit length
            embedding = embedding / np.sqrt(embedding.dot(embedding))
            elasticsearch_payload.append({'_index': 'wordEmbedding', '_id': word, '_source': {'embedding': embedding.tolist()}})

        bulk(self.es, elasticsearch_payload)

    def semantic_distance(self, v1, v2):
        if v1 is None or v2 is None:
            print("unknowns")
            return -99
        else:
            v1 = np.array(v1[0])
            v2 = np.array(v2[0])
            sim = np.dot(v1, v2.T)
        return sim

    def get_embedding_for_word(self, word):

        try:
            doc = self.es.get(index='wordEmbedding', id=word)
            return doc['_source']['embedding']
        except NotFoundError:
            return None

    def get_distance_between_column_labels(self, l1, l2):
        l1_tokens = l1.split(' ')
        l2_tokens = l2.split(' ')
        if l1 == l2:
            return 1.0
        if l1 == '' and l2 != '' or l1 != '' and l2 == '':
            return 0.0
        intersection = set(l1_tokens).intersection(l2_tokens)
        # remove common token
        if len(l1_tokens) > 1 and len(l2_tokens) > 1:
            for t in intersection:
                l1_tokens.remove(t)
                l2_tokens.remove(t)
        if len(l1_tokens) >= 1 and len(l2_tokens) == 0 or len(l2_tokens) >= 1 and len(l1_tokens) == 0:
            l1_tokens = l1.split(' ')
            l2_tokens = l2.split(' ')
        combinations = list(itertools.product(l1_tokens, l2_tokens))
        distance = 0.0
        for v1, v2 in combinations:
            emb1 = self.get_embedding_for_mwe(v1)
            emb2 = self.get_embedding_for_mwe(v2)
            if emb1[0] is None or emb2[0] is None:
                return 0
            d = self.semantic_distance(emb1, emb2)
            distance += d
        return (distance / len(combinations)) if combinations else 1

    def get_embedding_for_mwe(self, mwe):
        # TODO: [Refactor] remove or rename this method
        words = mwe.strip().split()
        mwe_vecs = []
        for w in words:
            mwe_vecs.append(self.get_embedding_for_word(w))

        return mwe_vecs


if __name__ == '__main__':
    wiki_model = WordEmbeddings()
    # wiki_model.load_vocab_to_elasticsearch('/home/mossad/projects/kglids/data_items/data/glove.6B.100d.txt')
    print(wiki_model.get_distance_between_column_labels('Net amount', 'Net Amount'))
    print("ss: " + str(wiki_model.semantic_distance(wiki_model.get_embedding_for_word("wife"),
                                                    wiki_model.get_embedding_for_word("spouse"))))
