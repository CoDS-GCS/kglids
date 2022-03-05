class Triplet:
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __repr__(self):
        return self.__repr_helper(True)

    def __repr_helper(self, isRoot):
        formattedSubject = self.subject
        formattedObject = self.object
        if isinstance(self.subject, Triplet):
            formattedSubject = '<<{}>>'.format(self.subject.__repr_helper(False))
        if isinstance(self.object, Triplet):
            formattedObject = '<<{}>>'.format(self.object.__repr_helper(False))

        if isRoot:
            return '{} {} {}.'.format(formattedSubject, self.predicate, formattedObject)
        else:
            return '{} {} {}'.format(formattedSubject, self.predicate, formattedObject)

    def __str__(self):
        return str(self.__repr__())
