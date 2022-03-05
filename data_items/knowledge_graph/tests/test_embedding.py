import sys

sys.path.insert(0, '../src')
from word_embedding.embeddings_client import n_similarity, get_embedding_of, get_similarity_between


# using wiki-news-300d-1M

def test_similarity():
    sim1 = n_similarity(['university'], ['school'])
    sim2 = n_similarity(['drink'], ['eat'])
    sim3 = n_similarity(['country'], ['mug'])
    assert (sim1 == 0.6891666545587082)
    assert (sim2 == 0.6586206951402682)
    assert (sim3 == 0.34181457410455424)


def test_get_embedding():
    drink_embedding = get_embedding_of('drink')
    assert (len(drink_embedding) == 300)
    not_existing_result = get_embedding_of('wndfownfpowfpo')
    assert (list(not_existing_result) == [])


def test_get_similarity_between():
    label_1 = 'Good morning'
    label_2 = 'Good afternoon'
    assert(get_similarity_between({'1': label_1, '2': label_2})[0][2] == 0.912544689683739)
    assert(get_similarity_between({'1': 'hello there', '2': ''})[0][2] == 0)
    assert(get_similarity_between({'1': '', '2': ''})[0][2] == 1.0)
    assert(get_similarity_between({'1': 'hello there', '2': 'hello there'})[0][2] == 1.0)