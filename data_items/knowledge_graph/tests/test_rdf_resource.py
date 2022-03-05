import sys

sys.path.insert(0, '../src')

from rdf_resource import RDFResource
from enums.relation import Relation
from label import Label


def test_rdfResource_str():
    namespace_relation = 'http://www.example.org/hlel#'
    namespace_column = 'http://www.example.org/hlel#'

    rdf_relation = RDFResource(Relation.cardinality.name, False, namespace_relation)
    rdf_col_node = RDFResource('4566', False, namespace_column)
    rdf_valueTXT = RDFResource('version')
    rdf_valueNUM = RDFResource(45)
    rdf_blank = RDFResource('entity', True)
    rdf_label = RDFResource(Label('test c1', 'en'))

    assert (str(rdf_relation) == '<http://www.example.org/hlel#cardinality>')
    assert (str(rdf_col_node) == '<http://www.example.org/hlel#4566>')
    assert (str(rdf_valueTXT) == '\"version\"')
    assert (str(rdf_valueNUM) == '45')
    assert (str(rdf_blank) == '_:entity')
    assert (str(rdf_label) == '\"test c1\"@en')
