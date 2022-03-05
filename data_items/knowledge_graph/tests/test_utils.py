import sys

sys.path.insert(0, '../src/')

from utils import generate_label


def test_generate_label():
    cs = generate_label('computerScience', 'en')
    countries = generate_label('countries.csv', 'en')
    random = generate_label('yooYoo_rrr965', 'fr')
    assert (cs.get_text() == 'computer science')
    assert (countries.get_text() == 'countries')
    assert (random.get_text() == 'yoo yoo rrr 965')
