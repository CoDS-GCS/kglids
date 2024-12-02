class CallComponents:
    package: str or list
    base_package: str
    file: str or None

    def __init__(self):
        self.package = ''
        self.base_package = ''
        self.file = None
        self.parent_library = ''
        self.rest = []

    def extract_parent_library(self):
        self.parent_library, *self.rest = self.package.split('.')

    def rewrite_library_path(self, package):
        if package is None:
            return
        if type(package) in (str, list, int):
            return

        self.base_package = self.parent_library
        self.package = f"{package.library_path}.{package.name}.{'.'.join(self.rest)}"


class CallArgumentsComponents:
    is_block: bool
    label: str
    keys: iter
    call_args: dict
    file_args: dict
    class_args: list

    def __init__(self, keys):
        self.is_block = False
        self.label = ''
        self.keys = iter(keys)
        self.call_args = {}
        self.file_args = {}
        self.class_args = []

    def next_label(self):
        self.label = next(self.keys, '')

    def set_is_block(self, parameters):
        self.is_block = '*' in self.label
        if self.is_block:
            parameters[self.label] = []


class AssignComponents:
    file: str or None
    variable: str or None
    value: str or list  # TODO: Refactor type to array

    def __init__(self):
        self.file = None
        self.variable = None
        self.value = ''


class BinOpComponents:
    left: str or None
    right: str or None

    def __init__(self):
        self.left = None
        self.right = None


class AttributeComponents:
    path: str or None
    file: str or None
    parent_path: str or None

    def __init__(self):
        self.path = None
        self.file = None
        self.parent_path = None


