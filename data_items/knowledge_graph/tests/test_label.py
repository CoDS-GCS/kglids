import sys

sys.path.insert(0, '../src')
from label import Label


def test_label():
    label = Label('test cs', 'en')
    assert (str(label) == '\"test cs\"@en')
