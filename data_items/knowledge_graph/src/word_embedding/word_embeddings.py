import numpy as np
import statistics
import itertools

class WordEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path
        self.w = None
        self.vocab = None
        self.ivocab = None
        self.vector_feature_size = 0

    def load_model(self):
        self.w, self.vocab, self.ivocab = self.load_vocab()

    def load_vocab(self):
        with open(self.model_path, 'r', encoding='utf-8') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        with open(self.model_path, 'r', encoding='utf-8') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]

        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        global vector_feature_size
        vector_feature_size = vector_dim
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            W[vocab[word], :] = v

        # normalize each word vector to unit variance
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T
        return (W_norm, vocab, ivocab)

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
        # print("vocab", )
        # print(len(vocab))
        if word in self.vocab:
            return self.w[self.vocab[word], :]
        else:
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

    def mwe_semantic_distance(self, vs1, vs2):
        cmp_count = 0
        sims = list()
        for v1 in vs1:
            for v2 in vs2:
                cmp_count += 1
                if v1 is None:
                    #TODO: use minimum edit distance
                    sims.append(0.0)
                    continue
                if v2 is None:
                    sims.append(0.0)
                    continue
                sim = np.dot(v1, v2.T)
                sims.append(sim)
        else:
            return statistics.mean(sims)

    def get_embedding_for_mwe(self, mwe):
        words = mwe.strip().split()
        mwe_vecs = list()

        for w in words:
            if w in self.vocab:
                mwe_vecs.append(self.w[self.vocab[w], :])
            else:
                mwe_vecs.append(None)
        else:
            return mwe_vecs


if __name__ == '__main__':
    wiki_model = WordEmbeddings(r'/home/disooqi/projects/wise/word_embedding/wiki-news-300d-1M.txt')
    wiki_model.load_model()
    print('Done loading')
    print("ss: " + str(wiki_model.semantic_distance(wiki_model.get_embedding_for_word("wife"),
                                                    wiki_model.get_embedding_for_word("spouse"))))
