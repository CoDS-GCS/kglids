from enum import Enum


class ColumnDataType(Enum):
    INT = 'int'
    FLOAT = 'float'
    BOOLEAN = 'boolean'
    DATE = 'date'
    NATURAL_LANGUAGE_NAMED_ENTITY = 'named_entity'
    NATURAL_LANGUAGE_TEXT = 'natural_language_text'
    STRING = 'string'
