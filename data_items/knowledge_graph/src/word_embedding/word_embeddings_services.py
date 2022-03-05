from word_embedding.word_embeddings import WordEmbeddings
import tqdm

class WordEmbeddingServices:

    def __init__(self):
        self.wiki_model = WordEmbeddings(r'../../../../embeddings/glove.6B.100d.txt')

        self.wiki_model.load_model()
        print('Done loading')

    def request_semantic_affinity(self, word_1, word_2):
        return self.wiki_model.semantic_distance(self.wiki_model.get_embedding_for_mwe(word_1),
                                                 self.wiki_model.get_embedding_for_mwe(word_2))

    def get_embedding_of(self, word):
        result = self.wiki_model.get_embedding_for_word(word)
        if result is None:
            return []
        return list(result)

    def get_affinity_between_column_names(self, d: dict):
        
        schemaSimilarities = []
        for c1_id, c1_label in d.items():
            for c2_id, c2_label in d.items():
                if c1_id == c2_id:
                    continue
                pair = (c1_id, c2_id, self.wiki_model.get_distance_between_column_labels(c2_label, c1_label))
                schemaSimilarities.append(pair)
        return schemaSimilarities

    def get_wiki_model(self):
        return self.wiki_model

