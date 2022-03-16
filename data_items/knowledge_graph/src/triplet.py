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