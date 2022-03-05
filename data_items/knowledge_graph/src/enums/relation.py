import enum


class Relation(enum.Enum):
    dataType = 1
    isPartOf = 2
    certainty = 3
    cardinality = 4
    semanticSimilarity = 5
    contentSimilarity = 6
    pkfk = 7
    inclusionDependency = 8
    deep_embeddings = 9
