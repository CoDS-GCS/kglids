import re
from camelsplit import camelsplit


class Label:

    def __init__(self, text: str, lan: str):
        self.text = text
        self.lan = lan

    def get_text(self) -> str:
        return self.text

    def get_lan(self) -> str:
        return self.lan

    def __repr__(self):
        return "\"" + self.text + "\"" + "@" + self.lan

    def __str__(self):
        return str(self.__repr__())


class RDFResource:
    def __init__(self, content, namespace=None, isBlank=False):
        self.content = content
        self.isBlank = isBlank
        self.namespace = namespace

    def __repr__(self):
        if self.namespace:
            return "<{}{}>".format(self.namespace, self.content)

        if self.isBlank:
            return '_:{}'.format(self.content)
        if isinstance(self.content, str):
            return '\"{}\"'.format(self.content)
        if isinstance(self.content, int):
            return '\"{}\"^^xsd:integer'.format(self.content)
        if isinstance(self.content, float):
            return '\"{}\"^^xsd:double'.format(round(self.content, 3)) 
        if isinstance(self.content, Label):
            return str(self.content)
        
        return self.content

    def __str__(self):
        return str(self.__repr__())


class Triplet:
    # TODO: [Refactor] Rename to RDFTriple
    def __init__(self, rdf_subject, rdf_predicate, rdf_object):
        self.rdf_subject = rdf_subject
        self.rdf_predicate = rdf_predicate
        self.rdf_object = rdf_object

    def __repr__(self):
        return self.__repr_helper(True)

    def __repr_helper(self, isRoot):
        formattedSubject = self.rdf_subject
        formattedObject = self.rdf_object
        if isinstance(self.rdf_subject, Triplet):
            formattedSubject = '<<{}>>'.format(self.rdf_subject.__repr_helper(False))
        if isinstance(self.rdf_object, Triplet):
            formattedObject = '<<{}>>'.format(self.rdf_object.__repr_helper(False))

        if isRoot:
            return '{} {} {}.'.format(formattedSubject, self.rdf_predicate, formattedObject)
        else:
            return '{} {} {}'.format(formattedSubject, self.rdf_predicate, formattedObject)

    def __str__(self):
        return str(self.__repr__())

    def get_reversed_triple(self):
        # TODO: [Refactor] a better name for this method?
        # The reverse of an RDF-star triple, is the reverse of the subject
        if isinstance(self.rdf_subject, Triplet):
            return Triplet(self.rdf_subject.get_reversed_triple(), self.rdf_predicate, self.rdf_object)
        
        return Triplet(self.rdf_object, self.rdf_predicate, self.rdf_subject)
    

def generate_label(col_name: str, lan: str) -> Label:
    # TODO: [Implement] the way labels are generated is not 100% the best. It is not always best to split by camel case
    if '.csv' in col_name:
        col_name = re.sub('.csv', '', col_name)
    col_name = re.sub('[^0-9a-zA-Z]+', ' ', col_name)
    text = " ".join(camelsplit(col_name.strip()))
    text = re.sub('\s+', ' ', text.strip())
    return Label(text.lower(), lan)
