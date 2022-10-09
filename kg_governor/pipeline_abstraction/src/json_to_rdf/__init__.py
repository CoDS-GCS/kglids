import json
from functools import reduce
from typing import List


def create_prefix() -> str:
    return '\n'.join([
        "@prefix pipeline: <http://kglids.org/ontology/pipeline/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix kglids: <http://kglids.org/ontology/> .",
        "\n"
    ])


def create_statement_uri(uri: str) -> str:
    return f'<{uri}> a kglids:Statement;\n'


def add_to_dict(dict, index, value):
    dict[index].append(value)
    return dict


def library_call_to_rdf(libraries: List[dict]) -> str:
    objects = reduce(lambda l, x: add_to_dict(l, x['call_type'], x['uri']), libraries, {
        'callsFunction': [],
        'callsClass': [],
        'callsPackage': [],
        'callsLibrary': [],
        'callsAPI': []
    })

    return ''.join([
        build_library_call_to_rdf(call_type, libs) for call_type, libs in objects.items()
    ])


def build_library_call_to_rdf(call_type: str, libraries: List[str]) -> str:
    if len(libraries) == 0:
        return ""
    return f'\tpipeline:{call_type} {", ".join([f"<{lib}>" for lib in libraries])};\n'


def read_to_rdf(reads: List[dict]) -> str:
    objects = reduce(lambda l, x: add_to_dict(l, x['type'], x['uri']), reads, {
        'readsTable': [],
        'readsColumn': [],
    })

    return ''.join([
        build_read_to_rdf(read_type, read) for read_type, read in objects.items()
    ])


def build_read_to_rdf(read_type: str, reads: List[dict]) -> str:
    if len(reads) == 0:
        return ""
    return f'\tpipeline:{read_type} {", ".join([f"<{read}>" for read in reads])};\n'


def has_text_to_rdf(text: str) -> str:
    return f'\tpipeline:hasText {escape_characters(text)};\n'


def has_parameter_to_rdf(parameters: List[dict]) -> str:
    if len(parameters) == 0:
        return ""
    return f'\tpipeline:hasParameter ' \
           f'{", ".join([create_quoted_parameter_value(param) for param in parameters])};\n'


def create_quoted_value(word: str) -> str:
    return json.dumps(word)


def create_quoted_parameter_value(param: dict) -> str:
    value = param['parameter']
    return create_quoted_value(value)


def has_dataflow_to_rdf(flows: List[str]) -> str:
    if len(flows) == 0:
        return ""
    return f'\tpipeline:hasDataFlowTo {", ".join([f"<{lib}>" for lib in flows])};\n'


def next_statement_to_rdf(statement: str or None) -> str:
    if statement is None:
        return ''
    return f'\tpipeline:hasNextStatement <{statement}>;\n'


def control_flow_to_rdf(control_flows: List[str]) -> str:
    if len(control_flows) == 0:
        return ""
    return f'\tpipeline:inControlFlow {", ".join([f"<{flow}>" for flow in control_flows])};\n'


def targets_to_rdf(targets: List[str]) -> str:
    if len(targets) == 0:
        return ""
    return f'\tpipeline:hasTarget ' \
           f'{", ".join([f"<{target}>" for target in targets])};\n'


def features_to_rdf(features: List[str]) -> str:
    if len(features) == 0:
        return ""
    return f'\tpipeline:hasFeature ' \
           f'{", ".join([f"<{feature}>" for feature in features])};\n'


def build_statement_rdf(statement: dict) -> str:
    return ''.join([
        ''.join([
            create_statement_uri(statement['uri']),
            library_call_to_rdf(statement['calls']),
            read_to_rdf(statement['read']),
            has_text_to_rdf(statement['text']),
            control_flow_to_rdf(statement['control_flow']),
            has_parameter_to_rdf(statement['parameters']),
            has_dataflow_to_rdf(statement['dataFlow']),
            features_to_rdf(statement['features']),
            targets_to_rdf(statement['targets']),
            next_statement_to_rdf(statement['next'])
        ]).rsplit(';', 1)[0],
        ' .\n'
    ])


def build_table_rdf(uri: str) -> str:
    return f'<{uri}> a kglids:Table .\n'


def build_column_rdf(table_uri: str, column_uri: str) -> str:
    return f'<{column_uri}> a kglids:Column;\n' \
           f'\tkglids:isPartOf <{table_uri}> .\n'


def build_parameter_rdf(statement_uri: str, parameter: str, parameter_value: str) -> str:
    return f'<<<{statement_uri}> pipeline:hasParameter {escape_characters(parameter)}>> pipeline:withParameterValue {escape_characters(parameter_value)} .\n'


def build_library_rdf(library: dict) -> str:
    return f"<{library['uri']}> a <{create_library_uri(library['type'])}> .\n"


def build_sub_library_rdf(library: dict, parent_uri: str) -> str:
    return f"<{library['uri']}> a <{create_library_uri(library['type'])}>;\n" \
           f"\tkglids:isPartOf <{parent_uri}> .\n"


def create_library_uri(uri: str or None):
    return 'http://kglids.org/ontology/API' if uri is None else uri


def build_pipeline_rdf(pipeline: dict) -> str:
    return ''.join([
        pipeline_uri_to_rdf(pipeline['uri']),
        title_to_rdf(pipeline['title']),
        author_to_rdf(pipeline['author']),
        votes_to_rdf(pipeline['votes']),
        date_to_rdf(pipeline['date']),
        tags_to_rdf(pipeline['tags']),
        source_to_rdf(pipeline['url']),
        score_to_rdf(pipeline['score']),
        dataset_to_rdf(pipeline['dataset'])
    ])


def pipeline_uri_to_rdf(uri: str):
    return f"<{uri}> a kglids:Pipeline;\n"


def escape_characters(word: str) -> str:
    return json.dumps(word)


def title_to_rdf(title: str) -> str:
    return f"\trdfs:label {escape_characters(title)};\n"


def author_to_rdf(author: str) -> str:
    return f"\tpipeline:isWrittenBy {escape_characters(author)};\n"


def votes_to_rdf(votes: int) -> str:
    return f"\tpipeline:hasVotes {votes};\n"


def date_to_rdf(date: str) -> str:
    return f"\tpipeline:isWrittenOn {escape_characters(date)};\n"


def tags_to_rdf(tags: List[str]) -> str:
    if len(tags) == 0:
        return ""
    return f'\tpipeline:hasTag {", ".join([create_quoted_value(tag) for tag in tags])};\n'


def source_to_rdf(source: str) -> str:
    return f"\tpipeline:hasSourceURL {escape_characters(source)};\n"


def score_to_rdf(score: float) -> str:
    return f"\tpipeline:hasScore {score};\n"


def dataset_to_rdf(dataset: str) -> str:
    return f"\tkglids:isPartOf <{dataset}> .\n"


def build_pipeline_rdf_page(statements: List[dict], datasets: List[dict]) -> str:
    return ''.join([
        create_prefix(),
        build_statement_rdf_part(statements),
        '\n',
        build_datasets_rdf_part(datasets)
    ])


def build_parameter_rdf_part(statement_uri: str, parameters: List[dict]):
    return '\n'.join([
        build_parameter_rdf(statement_uri, param['parameter'], param['parameter_value']) for param in parameters
    ])


def build_statement_rdf_part(statements: List[dict]) -> str:
    return '\n'.join(['\n'.join([
        build_statement_rdf(statement),
        build_parameter_rdf_part(statement['uri'], statement['parameters'])
    ]) for statement in statements])


def build_column_rdf_part(parent: str, columns: List[dict]) -> str:
    return '\n'.join([build_column_rdf(parent, column['uri']) for column in columns])


def build_datasets_rdf_part(datasets: List[dict]) -> str:
    return '\n'.join('\n'.join([
        build_table_rdf(table['uri']),
        build_column_rdf_part(table['uri'], table['contain'])
    ]) for table in datasets)


def build_library_rdf_page(libraries: List[dict]) -> str:
    return ''.join([
        create_prefix(),
        build_library_part(libraries)
    ])


def build_library_part(libraries: List[dict]) -> str:
    return '\n'.join(['\n'.join([
        build_library_rdf(library),
        build_sub_libraries_part(library['uri'], library['contain'])
    ]) for library in libraries])


def build_sub_libraries_part(parent_library: str, sub_libraries: List[dict]) -> str:
    return ''.join(['\n'.join([
        build_sub_library_rdf(library, parent_library),
        build_sub_libraries_part(library['uri'], library['contain'])
    ]) for library in sub_libraries])


def build_default_rdf_page(pipelines: List[dict]) -> str:
    return ''.join([
        create_prefix(),
        build_pipeline_part(pipelines),
        "\n"
    ])


def build_pipeline_part(pipelines: List[dict]) -> str:
    return '\n'.join([
        build_pipeline_rdf(pipeline) for pipeline in pipelines
    ])
