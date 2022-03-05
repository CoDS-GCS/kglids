from label import Label


class RDFResource:
    def __init__(self, content, isBlank=False, namespace=None):
        self.content = content
        self.isBlank = isBlank
        self.namespace = namespace

    def __repr__(self):
        if self.namespace:
            return "<{}{}>".format(self.namespace, self.content)
        else:
            if self.isBlank:
                return '_:{}'.format(self.content)
            if isinstance(self.content, str):
                return '\"{}\"'.format(self.content)
            if isinstance(self.content, Label):
                return str(self.content)
            return self.content

    def __str__(self):
        return str(self.__repr__())
