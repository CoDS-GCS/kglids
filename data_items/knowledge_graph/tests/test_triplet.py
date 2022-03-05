import sys

sys.path.insert(0, '../src/')

from enums.relation import Relation

from rdf_resource import RDFResource
from triplet import Triplet

namespace = 'http://www.example.org/hlel#'


def test_triplet():
    subject = RDFResource('from', False, namespace)
    predicate = RDFResource(Relation.cardinality.name, False, namespace)
    object_col = RDFResource('to', False, namespace)
    object_literal_txt = RDFResource('version')
    object_literal_num = RDFResource(45)

    ref_to_ref = Triplet(subject, predicate, object_col)
    ref_to_txt = Triplet(subject, predicate, object_literal_txt)
    ref_to_num = Triplet(subject, predicate, object_literal_num)

    actual_ref_to_ref = '<http://www.example.org/hlel#from> <http://www.example.org/hlel#cardinality> ' \
                        '<http://www.example.org/hlel#to>.'
    actual_ref_to_txt = '<http://www.example.org/hlel#from> <http://www.example.org/hlel#cardinality> ' \
                        '\"version\".'
    actual_ref_to_num = '<http://www.example.org/hlel#from> <http://www.example.org/hlel#cardinality> ' \
                        '45.'

    assert (str(ref_to_ref) == actual_ref_to_ref)
    assert (str(ref_to_txt) == actual_ref_to_txt)
    assert (str(ref_to_num) == actual_ref_to_num)


def test_nested_triplets():
    # Nested subject
    nested_subject = RDFResource('from', False, namespace)
    nested_predicate = RDFResource(Relation.contentSimilarity.name, False, namespace)
    nested_object = RDFResource('to', False, namespace)

    subject = Triplet(nested_subject, nested_predicate, nested_object)
    predicate = RDFResource(Relation.certainty.name, False, namespace)
    objct = RDFResource(0.5)

    final_triplet = Triplet(subject, predicate, objct)
    actual_Triplet = '<<<http://www.example.org/hlel#from> ' \
                     '<http://www.example.org/hlel#contentSimilarity> ' \
                     '<http://www.example.org/hlel#to>>> ' \
                     '<http://www.example.org/hlel#certainty> ' \
                     '0.5.'
    assert (str(final_triplet) == actual_Triplet)
