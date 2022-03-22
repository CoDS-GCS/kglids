import numpy as np
import statistics
import itertools

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class WordEmbeddings:
    def __init__(self, endpoint='http://localhost:9200'):
        self.es = Elasticsearch(endpoint)

    def load_vocab_to_elasticsearch(self, path_to_raw_embeddings):
        if self.es.indices.exists(index='word_embedding'):
            print('Word embedding index exists in elasticsearch')
            return
        
        with open(path_to_raw_embeddings, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f'Loading {len(lines)} word embeddings into elastic')
        
        elasticsearch_payload = []
        for line in lines:
            split_line = line.strip().split()
            word = split_line[0]
            embedding = np.array([float(i) for i in split_line[1:]])
            # normalize to unit length
            embedding = embedding / np.sqrt(embedding.dot(embedding))
            elasticsearch_payload.append({'_index': 'word_embedding', '_id': word, '_source': {'embedding': embedding.tolist()}})
        self.es.index(index='word_embedding', id='hello', body={})
        bulk(self.es, elasticsearch_payload)

    def semantic_distance(self, v1, v2):
        if v1 is None or v2 is None:
            print("unknowns")
            return -99
        else:
            v1 = np.array(v1)
            v2 = np.array(v2)
            sim = np.dot(v1, v2.T)
        return sim

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
            l1_tokens = [i for i in l1_tokens if i not in intersection]
            l2_tokens = [i for i in l2_tokens if i not in intersection]
            
        if len(l1_tokens) >= 1 and len(l2_tokens) == 0 or len(l2_tokens) >= 1 and len(l1_tokens) == 0:
            l1_tokens = l1.split(' ')
            l2_tokens = l2.split(' ')
        word_embeddings_for_tokens = self.get_embeddings_for_tokens(l1_tokens+l2_tokens)
        
        combinations = list(itertools.product(l1_tokens, l2_tokens))
        distance = 0.0
        for v1, v2 in combinations:
            emb1 = word_embeddings_for_tokens[v1]
            emb2 = word_embeddings_for_tokens[v2]
            if emb1 is None or emb2 is None:
                return 0
            
            d = self.semantic_distance(emb1, emb2)
            distance += d
        return (distance / len(combinations)) if combinations else 1
    
    
    def get_embeddings_for_tokens(self, tokens):
        embeddings = {}
        # TODO: [Implement] does it make sense to take only the first taken?
        req_body = {'ids': [i.strip().split()[0] for i in tokens]}  
        res = self.es.mget(index='word_embedding', body=req_body)
        for i in range(len(tokens)):
            embeddings[tokens[i]] = res['docs'][i]['_source']['embedding'] if res['docs'][i]['found'] else None
        return embeddings
    