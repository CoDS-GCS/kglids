import os
import itertools
import pickle

import numpy as np


# TODO [Implement] figure out a better way than just loading the word embeddings to memory
class WordEmbeddings:
    def __init__(self, word_embeddings_path):
        self.raw_word_embeddings_path = word_embeddings_path
        self.normalized_word_embeddings_path = self.raw_word_embeddings_path[:self.raw_word_embeddings_path.rindex('.')] + '.pickle'
        # initialize and normalize the word vectors if they don't exist
        if not os.path.exists(self.normalized_word_embeddings_path):
            self._initialize_and_normalize_word_embeddings()
        
        with open(self.normalized_word_embeddings_path, 'rb') as f:
            print('Loading:', self.normalized_word_embeddings_path)
            self.vectors = pickle.load(f)
    
    def _initialize_and_normalize_word_embeddings(self):
        # loads raw word embedding file, normalizes the vectors to unit length, and stores them as pickle
        print('initializing raw word embeddings and normalizing them to unit length')
        with open(self.raw_word_embeddings_path, 'r') as f:
            lines = [i.strip() for i in f.readlines()]
        vectors = {}
        for line in lines:
            split = line.split()
            word = split[0]
            vector = np.array([float(i) for i in split[1:]])
            vector /= np.linalg.norm(vector)                 # normalize to unit length
            vectors[word] = vector
        with open(self.normalized_word_embeddings_path, 'wb') as f:
            pickle.dump(vectors, f)
        
            
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
        word_embeddings_for_tokens = self.get_embeddings_for_tokens(l1_tokens + l2_tokens)

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
        # TODO: [Implement] does it make sense to take only the first taken?
        embeddings = {i: self.vectors.get(i.strip().split()[0], None) for i in tokens}
        return embeddings

