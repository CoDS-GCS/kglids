import abc

from neo4j import GraphDatabase
from neo4j.exceptions import CypherTypeError

uri = "neo4j://localhost:7687"


class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def clean_up(self):
        raise NotImplementedError


class Neo4JRepository(AbstractRepository):
    def __init__(self, python_file_name: str, source: str, dataset_name: str):
        self.session = GraphDatabase.driver(uri,
                                            auth=("neo4j", "123456")
                                            ).session()
        self.python_file_name = python_file_name
        self.source = source
        self.dataset_name = dataset_name

    def clean_up(self):
        self.session.run(
            "MATCH (p:Parameters) "
            "WHERE NOT (p)-[]->() "
            "DETACH DELETE p"
        )

    def delete_everything(self):
        self.session.run(
            "MATCH (p) "
            "DETACH DELETE p"
        )
        self.session.run("call n10s.graphconfig.drop()")

    def database_init(self):
        mapping = [
            {'neoSchema': 'IS_PART_OF', 'kglidsSchema': 'http://kglids.org/ontology/isPartOf'},
            {'neoSchema': 'MANIPULATE', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/reads'},
            {'neoSchema': 'NEXT', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasNextStatement'},
            {'neoSchema': 'FLOWS_TO', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasDataFlowTo'},
            {'neoSchema': 'calls', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/callsLibrary'},
            {'neoSchema': 'hasParameter', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasParameter'},
            {'neoSchema': 'parameterValue', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/withParameterValue'},
            {'neoSchema': 'control_flow', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/inControlFlow'},
            {'neoSchema': 'hasText', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasText'},
            {'neoSchema': 'author', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/isWrittenBy'},
            {'neoSchema': 'date', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/isWrittenOn'},
            {'neoSchema': 'url', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasSourceURL'},
            {'neoSchema': 'score', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasScore'},
            {'neoSchema': 'votes', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasVotes'},
            {'neoSchema': 'tags', 'kglidsSchema': 'http://kglids.org/ontology/pipeline/hasTag'},
            {'neoSchema': 'title', 'kglidsSchema': 'http://www.w3.org/2000/01/rdf-schema#label'},
            {'neoSchema': 'Source', 'kglidsSchema': 'http://kglids.org/ontology/Source'},
            {'neoSchema': 'Dataset', 'kglidsSchema': 'http://kglids.org/ontology/Dataset'},
            {'neoSchema': 'Table', 'kglidsSchema': 'http://kglids.org/ontology/Table'},
            {'neoSchema': 'Column', 'kglidsSchema': 'http://kglids.org/ontology/Column'},
            {'neoSchema': 'Pipeline', 'kglidsSchema': 'http://kglids.org/ontology/Pipeline'},
            {'neoSchema': 'Library', 'kglidsSchema': 'http://kglids.org/ontology/Library'},
            {'neoSchema': 'Statement', 'kglidsSchema': 'http://kglids.org/ontology/Statement'}
        ]
        uris = [
            {'prefix': 'kglids', 'uri': 'http://kglids.org/ontology/'},
            {'prefix': 'rdfs', 'uri': 'http://www.w3.org/2000/01/rdf-schema#'},
            {'prefix': 'pipeline', 'uri': 'http://kglids.org/ontology/pipeline/'},
        ]

        self.session.run("""
        WITH $uri as uris
        CALL n10s.graphconfig.init({handleVocabUris: "MAP"}) YIELD param
        UNWIND uris as u
        CALL n10s.nsprefixes.add(u.prefix, u.uri) YIELD namespace
        RETURN count(namespace)
        """, uri=uris)
        self.session.run("""
        WITH $mapping as mappings 
        UNWIND mappings AS m
        CALL n10s.mapping.add(m.kglidsSchema,m.neoSchema) YIELD schemaElement
        RETURN COUNT(schemaElement)
        """, mapping=mapping)

    def create_file_nodes(self, files):
        self.session.run("FOREACH (table in $files | "
                         "CREATE (f:Resource:Table {uri: table.uri}) "
                         "FOREACH (column in table.contain | "
                         "MERGE (c:Resource:Column {uri: column.uri}) "
                         "MERGE (c)-[:IS_PART_OF]->(f)"
                         ")"
                         ")",
                         files=files)

    def create_library_nodes(self, libraries):
        self.session.run("FOREACH (library in $libraries | "
                         "CREATE (main:Resource:Library {uri: library.uri}) "
                         "FOREACH (sublib in library.contain | "
                         "CREATE (sub:Resource:Library {uri: sublib.uri}) "
                         "MERGE (sub)-[:IS_PART_OF]->(main) "
                         "FOREACH (subsublib in sublib.contain | "
                         "CREATE (s:Resource:Library {uri: subsublib.uri}) "
                         "MERGE (s)-[:IS_PART_OF]->(sub)"
                         ")"
                         ")"
                         ")",
                         libraries=libraries)

    def create_first_node(self, node):
        self.session.run("FOREACH (b in $nodes | "
                         "CREATE (n:Resource:Statement { uri: b.uri, hasText: b.text, calls: b.calls }) "
                         "FOREACH (flow in b.control_flow | "
                         "MERGE (f:Resource {uri: flow}) "
                         "MERGE (n)-[:control_flow]->(f) "
                         ") "
                         "FOREACH (parameter in b.parameters | "
                         "CREATE (p:Literal {value: parameter.parameter}) "
                         "MERGE (n)-[:hasParameter {parameterValue: parameter.parameter_value}]->(p) "
                         ") "
                         "FOREACH (element in b.read | "
                         "MERGE (c:Resource {uri: element.uri}) "
                         "MERGE (n)-[:MANIPULATE]->(c) "
                         "))",
                         nodes=[node])

    def create_node_nodes(self, nodes):
        try:
            self.session.run("WITH $nodes as nodes "
                             "FOREACH (b in nodes | "
                             "CREATE (n:Resource:Statement { uri: b.uri, hasText: b.text, calls: b.calls }) "
                             "MERGE (prev:Resource {uri: b.previous}) "
                             "MERGE (prev)-[:NEXT]->(n) "
                             "FOREACH (flow in b.control_flow | "
                             "MERGE (f:Resource {uri: flow}) "
                             "MERGE (n)-[:control_flow]->(f) "
                             ") "
                             "FOREACH (parameter in b.parameters | "
                             "CREATE (p:Literal {value: parameter.parameter}) "
                             "MERGE (n)-[:hasParameter {parameterValue: parameter.parameter_value}]->(p) "
                             ") "
                             "FOREACH (element in b.read | "
                             "MERGE (c:Resource {uri: element.uri}) "
                             "MERGE (n)-[:MANIPULATE]->(c) "
                             "))",
                             nodes=nodes)  # TODO: VERIFY THAT THIS WORKS
            self.session.run("FOREACH (b in $nodes | "
                             "MERGE (n:Resource:Statement { uri: b.uri })"
                             "FOREACH (data in b.dataFlow | "
                             "MERGE (r:Resource {uri: data})"
                             "MERGE (n)-[:FLOWS_TO]->(r)"
                             "))", nodes=nodes)
        except CypherTypeError as e:
            print(e)

    def create_pipeline_node(self, pipeline):
        self.session.run(
            """
            CREATE (:Pipeline {
                uri: $uri,
                author: $author,
                url: $url,
                date: $date,
                score: $score,
                votes: $votes,
                tags: $tags,
                title: $title,
                IS_PART_OF: $dataset
            })
            """,
            uri=pipeline['uri'],
            author=pipeline['author'],
            url=pipeline['url'],
            date=pipeline['date'],
            score=pipeline['score'],
            votes=pipeline['votes'],
            tags=pipeline['tags'],
            title=pipeline['title'],
            dataset=pipeline['dataset'])
