class Call:
    name = ''  # name of the function/class
    library_path = ''  # the library import path. e.g. for sklearn.svm.SVC it will be sklearn.svm
    parameters = {}  # contains the names and default values for the first 5 params
    is_class_def = None  # whether this Call is a class (or function)
    return_types = []  # the return types of this call. For classes, the same object is returned
    is_relevant = True  # whether this call is relevant to the analysis (e.g. plotting functions aren't)

    def __init__(self, name, library_path, parameters, is_class_def, return_types=None, is_relevant=True):
        self.name = name
        self.library_path = library_path
        self.parameters = parameters
        self.is_class_def = is_class_def
        if self.is_class_def:
            self.return_types = [self]
        else:
            self.return_types = return_types


class File:
    __slots__ = ['id', 'filename', 'path']

    def __init__(self, id, filename, path):
        self.id = id
        self.filename = filename
        self.path = path