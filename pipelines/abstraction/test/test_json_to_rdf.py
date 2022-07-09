import unittest
from src.json_to_rdf import (create_prefix, library_call_to_rdf, read_to_rdf, has_text_to_rdf,
                             has_parameter_to_rdf, has_dataflow_to_rdf, next_statement_to_rdf,
                             create_statement_uri, control_flow_to_rdf, build_statement_rdf,
                             build_table_rdf, build_column_rdf, build_parameter_rdf, build_library_rdf,
                             build_sub_library_rdf, build_pipeline_rdf, build_pipeline_rdf_page, build_library_rdf_page,
                             build_default_rdf_page, title_to_rdf)


class MyTestCase(unittest.TestCase):
    def test_create_prefix_creates_prefix_on_newlines(self):
        expected = "@prefix pipeline: <http://kglids.org/ontology/pipeline/> .\n" \
                   "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n" \
                   "@prefix kglids: <http://kglids.org/ontology/> .\n\n"

        result = create_prefix()

        self.assertEqual(expected, result)

    def test_create_statement_uri_creates_uri_with_type(self):
        expected = "<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s8> " \
                   "a kglids:Statement;\n"
        uri = 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s8'

        result = create_statement_uri(uri)

        self.assertEqual(expected, result)

    def test_library_call_to_rdf_when_no_library_return_empty_string(self):
        expected = ""
        libraries = []

        result = library_call_to_rdf(libraries)

        self.assertEqual(expected, result)

    def test_library_call_to_rdf_when_one_library_return_correct_call_without_separator(self):
        expected = "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>;\n"
        libraries = [{'uri': 'http://kglids.org/resource/library/pandas/read_csv', 'call_type': 'callsFunction'}]

        result = library_call_to_rdf(libraries)

        self.assertEqual(expected, result)

    def test_library_call_to_rdf_when_more_library_return_correct_call_with_separator(self):
        expected = "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>;\n" \
                   "\tpipeline:callsClass <http://kglids.org/resource/library/pandas/DataFrame>;\n"
        libraries = [{'uri': 'http://kglids.org/resource/library/pandas/read_csv', 'call_type': 'callsFunction'},
                     {'uri': 'http://kglids.org/resource/library/pandas/DataFrame', 'call_type': 'callsClass'}]

        result = library_call_to_rdf(libraries)

        self.assertEqual(expected, result)

    def test_library_call_to_rdf_when_even_more_library_return_correct_call_with_separator(self):
        expected = "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>, <http://kglids.org/resource/library/pandas/read_csv>;\n" \
                   "\tpipeline:callsClass <http://kglids.org/resource/library/pandas/DataFrame>, <http://kglids.org/resource/library/pandas/DataFrame>;\n"
        libraries = [{'uri': 'http://kglids.org/resource/library/pandas/read_csv', 'call_type': 'callsFunction'},
                     {'uri': 'http://kglids.org/resource/library/pandas/DataFrame', 'call_type': 'callsClass'},
                     {'uri': 'http://kglids.org/resource/library/pandas/read_csv', 'call_type': 'callsFunction'},
                     {'uri': 'http://kglids.org/resource/library/pandas/DataFrame', 'call_type': 'callsClass'}]

        result = library_call_to_rdf(libraries)

        self.assertEqual(expected, result)

    def test_read_to_rdf_when_no_reads_return_empty_string(self):
        expected = ""
        reads = []

        result = read_to_rdf(reads)

        self.assertEqual(expected, result)

    def test_read_to_rdf_when_one_read_return_correct_call_without_separator(self):
        expected = "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n"
        reads = [
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                'type': 'readsTable'
            }]

        result = read_to_rdf(reads)

        self.assertEqual(expected, result)

    def test_reads_to_rdf_when_more_read_return_correct_call_with_separator(self):
        expected = "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n" \
                   "\tpipeline:readsColumn <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column>;\n"
        reads = [
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                'type': 'readsTable'},
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column',
                'type': 'readsColumn'}]

        result = read_to_rdf(reads)

        self.assertEqual(expected, result)

    def test_reads_to_rdf_when_more_read_return_correct_call_with_separator(self):
        expected = "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>, <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n" \
                   "\tpipeline:readsColumn <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column>, <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column>;\n"
        reads = [
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                'type': 'readsTable'
            },
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column',
                'type': 'readsColumn'
            },
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/column',
                'type': 'readsColumn'
            },
            {
                'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                'type': 'readsTable'
            }
        ]

        result = read_to_rdf(reads)

        self.assertEqual(expected, result)

    def test_hast_text_to_rdf_returns_text_when_called(self):
        expected = "\tpipeline:hasText \"raw_data = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')\";\n"
        text = "raw_data = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')"

        result = has_text_to_rdf(text)

        self.assertEqual(expected, result)

    def test_has_parameter_to_rdf_when_no_paramters_return_empty_string(self):
        expected = ""
        parameters = []

        result = has_parameter_to_rdf(parameters)

        self.assertEqual(expected, result)

    def test_has_parameter_to_rdf_when_one_parameter_return_correct_call_without_separator(self):
        expected = '\tpipeline:hasParameter "filepath_or_buffer";\n'
        parameters = [{"parameter": "filepath_or_buffer", "parameter_value": "filename"}]

        result = has_parameter_to_rdf(parameters)

        self.assertEqual(expected, result)

    def test_has_parameter_to_rdf_when_more_parameters_return_correct_call_with_separator(self):
        expected = '\tpipeline:hasParameter "filepath_or_buffer", "sep";\n'
        parameters = [
            {"parameter": "filepath_or_buffer", "parameter_value": "filename"},
            {"parameter": "sep", "parameter_value": ","}
        ]

        result = has_parameter_to_rdf(parameters)

        self.assertEqual(expected, result)

    def test_has_dataflow_to_rdf_when_no_flow_return_empty_string(self):
        expected = ""
        flows = []

        result = has_dataflow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_has_dataflow_to_rdf_when_one_flow_return_correct_call_without_separator(self):
        expected = "\tpipeline:hasDataFlowTo <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9>;\n"
        flows = [
            'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9']

        result = has_dataflow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_has_dataflow_to_rdf_when_more_flow_return_correct_call_with_separator(self):
        expected = "\tpipeline:hasDataFlowTo <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9>, " \
                   "<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9>;\n"
        flows = [
            'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9',
            'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9'
        ]

        result = has_dataflow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_control_flow_to_rdf_when_no_flow_return_empty_string(self):
        expected = ""
        flows = []

        result = control_flow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_control_flow_to_rdf_when_one_flow_return_correct_call_without_separator(self):
        expected = "\tpipeline:inControlFlow <http://kglids.org/resource/import>;\n"
        flows = ["http://kglids.org/resource/import"]

        result = control_flow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_control_flow_to_rdf_when_more_flow_return_correct_call_with_separator(self):
        expected = "\tpipeline:inControlFlow <http://kglids.org/resource/import>, <http://kglids.org/resource/import>;\n"
        flows = [
            "http://kglids.org/resource/import",
            "http://kglids.org/resource/import"
        ]

        result = control_flow_to_rdf(flows)

        self.assertEqual(expected, result)

    def test_next_statement_to_rdf_returns_text_when_called(self):
        expected = "\tpipeline:hasNextStatement <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9>;\n"
        statement = "http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s9"

        result = next_statement_to_rdf(statement)

        self.assertEqual(expected, result)

    def test_next_statement_to_rdf_returns_empty_string_when_no_next(self):
        expected = ""
        statement = None

        result = next_statement_to_rdf(statement)

        self.assertEqual(expected, result)

    def test_build_statement_rdf_returns_statement_text(self):
        expected = "<http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18> a kglids:Statement;\n" \
                   "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>;\n" \
                   "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n" \
                   "\tpipeline:hasText \"df = pd.read_csv(filename, encoding=\'ISO-8859-1\')\";\n" \
                   "\tpipeline:inControlFlow <http://kglids.org/resource/import>;\n" \
                   "\tpipeline:hasParameter \"filepath_or_buffer\";\n" \
                   "\tpipeline:hasDataFlowTo <http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19>;\n" \
                   "\tpipeline:hasNextStatement <http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19> .\n"

        statement = {
            "uri": "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18",
            "next": "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19",
            "text": "df = pd.read_csv(filename, encoding='ISO-8859-1')",
            "control_flow": [
                "http://kglids.org/resource/import"
            ],
            "parameters": [
                {"parameter": "filepath_or_buffer", "parameter_value": "filename"},
            ],
            "calls": [{'uri': "http://kglids.org/resource/library/pandas/read_csv", 'call_type': "callsFunction"}],
            "read": [
                {
                    'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                    'type': 'readsTable'}
            ],
            "dataFlow": [
                "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19",
            ]
        }

        result = build_statement_rdf(statement)
        self.assertEqual(expected, result)

    def test_build_statement_rdf_without_next_returns_statement_text_with_closing_tag(self):
        expected = "<http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18> a kglids:Statement;\n" \
                   "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>;\n" \
                   "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n" \
                   "\tpipeline:hasText \"df = pd.read_csv(filename, encoding=\'ISO-8859-1\')\";\n" \
                   "\tpipeline:inControlFlow <http://kglids.org/resource/import>;\n" \
                   "\tpipeline:hasParameter \"filepath_or_buffer\";\n" \
                   "\tpipeline:hasDataFlowTo <http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19> .\n"
        statement = {
            "uri": "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18",
            "next": None,
            "text": "df = pd.read_csv(filename, encoding='ISO-8859-1')",
            "control_flow": [
                "http://kglids.org/resource/import"
            ],
            "parameters": [
                {"parameter": "filepath_or_buffer", "parameter_value": "filename"},
            ],
            "calls": [{'uri': "http://kglids.org/resource/library/pandas/read_csv", 'call_type': "callsFunction"}],
            "read": [
                {
                    'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                    'type': 'readsTable'}
            ],
            "dataFlow": [
                "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19",
            ]
        }

        result = build_statement_rdf(statement)

        self.assertEqual(expected, result)

    def test_build_table_rdf_returns_string(self):
        expected = '<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv> a kglids:Table .\n'
        table_uri = 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv'

        result = build_table_rdf(table_uri)

        self.assertEqual(expected, result)

    def test_build_column_rdf_returns_string_with_link_to_table_rdf_type(self):
        expected = '<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/Time%20Occurred> a kglids:Column;\n' \
                   '\tkglids:isPartOf <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv> .\n'
        table_uri = 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv'
        column_uri = 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/Time%20Occurred'

        result = build_column_rdf(table_uri, column_uri)

        self.assertEqual(expected, result)

    def test_build_parameter_rdf_returns_rdf_with_parameter_and_parameter_value_linked(self):
        expected = "<<<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s8> pipeline:hasParameter \"sep\">> pipeline:withParameterValue \",\" .\n"
        statement_uri = "http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/arnehuang.la-traffic-data-eda/s8"
        parameter = 'sep'
        parameter_value = ','

        result = build_parameter_rdf(statement_uri, parameter, parameter_value)

        self.assertEqual(expected, result)

    def test_build_library_rdf_returns_library_with_correct_type(self):
        expected = "<http://kglids.org/resource/library/numpy> a <http://kglids.org/ontology/Library> .\n"
        library = {
            "uri": "http://kglids.org/resource/library/numpy",
            "contain": [],
            "type": "http://kglids.org/ontology/Library"
        }

        result = build_library_rdf(library)

        self.assertEqual(expected, result)

    def test_build_library_rdf_returns_function_with_correct_type_and_link_to_parent(self):
        expected = "<http://kglids.org/resource/library/numpy/array> a <http://kglids.org/ontology/Function>;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/library/numpy> .\n"
        library = {
            "uri": "http://kglids.org/resource/library/numpy/array",
            "contain": [],
            "type": "http://kglids.org/ontology/Function"
        }

        result = build_sub_library_rdf(library, 'http://kglids.org/resource/library/numpy')

        self.assertEqual(expected, result)

    def test_build_library_rdf_returns_library_with_parent_type_when_type_is_unknown(self):
        expected = "<http://kglids.org/resource/library/numpy> a <http://kglids.org/ontology/API> .\n"
        library = {
            "uri": "http://kglids.org/resource/library/numpy",
            "contain": [],
            "type": None
        }

        result = build_library_rdf(library)

        self.assertEqual(expected, result)

    def test_build_pipeline_default_object_with_correct_information(self):
        expected = "<http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm> a kglids:Pipeline;\n" \
                   "\trdfs:label \"NHL player salary Prediction - XGBoost, rf and SVM\";\n" \
                   "\tpipeline:isWrittenBy \"Cam Nugent\";\n" \
                   "\tpipeline:hasVotes 4;\n" \
                   "\tpipeline:isWrittenOn \"2017-09-22 20:18:19\";\n" \
                   "\tpipeline:hasTag \"arts and entertainment\";\n" \
                   "\tpipeline:hasSourceURL \"https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm\";\n" \
                   "\tpipeline:hasScore 0.632134442843743;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries> .\n"
        pipeline = {
            "url": "https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm",
            "title": "NHL player salary Prediction - XGBoost, rf and SVM",
            "author": "Cam Nugent",
            "votes": 4,
            "score": 0.632134442843743,
            "date": "2017-09-22 20:18:19",
            "tags": ["arts and entertainment"],
            "uri": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm",
            "dataset": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries"
        }

        result = build_pipeline_rdf(pipeline)

        self.assertEqual(expected, result)

    def test_build_pipeline_rdf_page_return_correct_page(self):
        expected = "@prefix pipeline: <http://kglids.org/ontology/pipeline/> .\n" \
                   "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n" \
                   "@prefix kglids: <http://kglids.org/ontology/> .\n\n" \
                   "<http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18> a kglids:Statement;\n" \
                   "\tpipeline:callsFunction <http://kglids.org/resource/library/pandas/read_csv>;\n" \
                   "\tpipeline:readsTable <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv>;\n" \
                   "\tpipeline:hasText \"df = pd.read_csv(filename, encoding=\'ISO-8859-1\')\";\n" \
                   "\tpipeline:inControlFlow <http://kglids.org/resource/import>;\n" \
                   "\tpipeline:hasParameter \"filepath_or_buffer\";\n" \
                   "\tpipeline:hasDataFlowTo <http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19>;\n" \
                   "\tpipeline:hasNextStatement <http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19> .\n\n" \
                   "<<<http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18> pipeline:hasParameter \"filepath_or_buffer\">> pipeline:withParameterValue \"filename\" .\n\n" \
                   "<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv> a kglids:Table .\n\n" \
                   "<http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/Time%20Occurred> a kglids:Column;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv> .\n"

        statements = [{
            "uri": "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s18",
            "next": "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19",
            "text": "df = pd.read_csv(filename, encoding='ISO-8859-1')",
            "control_flow": [
                "http://kglids.org/resource/import"
            ],
            "parameters": [
                {"parameter": "filepath_or_buffer", "parameter_value": "filename"},
            ],
            "calls": [{'uri': "http://kglids.org/resource/library/pandas/read_csv", 'call_type': 'callsFunction'}],
            "read": [
                {
                    'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
                    'type': 'readsTable'}
            ],
            "dataFlow": [
                "http://kglids.org/resource/kaggle/leonardopena.top50spotify2019/deepakdeepu8978.how-popular-a-song-is-according-to-spotify/s19",
            ]
        }]
        datasets = [{
            'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv',
            'contain': [
                {
                    'uri': 'http://kglids.org/resource/kaggle/cityofLA.los-angeles-traffic-collision-data/traffic-collision-data-from-2010-to-present.csv/Time%20Occurred',
                    'contain': []
                }
            ]
        }]

        result = build_pipeline_rdf_page(statements, datasets)

        self.assertEqual(expected, result)

    def test_build_library_rdf_page(self):
        expected = "@prefix pipeline: <http://kglids.org/ontology/pipeline/> .\n" \
                   "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n" \
                   "@prefix kglids: <http://kglids.org/ontology/> .\n\n" \
                   "<http://kglids.org/resource/library/numpy> a <http://kglids.org/ontology/Library> .\n\n" \
                   "<http://kglids.org/resource/library/numpy/array> a <http://kglids.org/ontology/Function>;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/library/numpy> .\n\n" \
                   "<http://kglids.org/resource/library/numpy/random> a <http://kglids.org/ontology/Package>;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/library/numpy> .\n\n" \
                   "<http://kglids.org/resource/library/numpy/random/seed> a <http://kglids.org/ontology/Function>;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/library/numpy/random> .\n\n" \
                   "<http://kglids.org/resource/library/numpy/random/seed/add> a <http://kglids.org/ontology/Function>;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/library/numpy/random/seed> .\n\n"

        libraries = [
            {
                "uri": "http://kglids.org/resource/library/numpy",
                "contain": [
                    {
                        "uri": "http://kglids.org/resource/library/numpy/array",
                        "contain": [],
                        "type": "http://kglids.org/ontology/Function"
                    },
                    {
                        "uri": "http://kglids.org/resource/library/numpy/random",
                        "contain": [
                            {
                                "uri": "http://kglids.org/resource/library/numpy/random/seed",
                                "contain": [
                                    {
                                        "uri": "http://kglids.org/resource/library/numpy/random/seed/add",
                                        "contain": [],
                                        "type": "http://kglids.org/ontology/Function"
                                    },
                                ],
                                "type": "http://kglids.org/ontology/Function"
                            },
                        ],
                        "type": "http://kglids.org/ontology/Package"
                    },
                ],
                "type": "http://kglids.org/ontology/Library"
            }]

        result = build_library_rdf_page(libraries)

        self.assertEqual(expected, result)

    def test_build_default_rdf_page(self):
        expected = "@prefix pipeline: <http://kglids.org/ontology/pipeline/> .\n" \
                   "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n" \
                   "@prefix kglids: <http://kglids.org/ontology/> .\n\n" \
                   "<http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm> a kglids:Pipeline;\n" \
                   "\trdfs:label \"NHL player salary Prediction - XGBoost, rf and SVM\";\n" \
                   "\tpipeline:isWrittenBy \"Cam Nugent\";\n" \
                   "\tpipeline:hasVotes 4;\n" \
                   "\tpipeline:isWrittenOn \"2017-09-22 20:18:19\";\n" \
                   "\tpipeline:hasTag \"arts and entertainment\";\n" \
                   "\tpipeline:hasSourceURL \"https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm\";\n" \
                   "\tpipeline:hasScore 0.632134442843743;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries> .\n\n" \
                   "<http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm> a kglids:Pipeline;\n" \
                   "\trdfs:label \"NHL player salary Prediction - XGBoost, rf and SVM\";\n" \
                   "\tpipeline:isWrittenBy \"Cam Nugent\";\n" \
                   "\tpipeline:hasVotes 4;\n" \
                   "\tpipeline:isWrittenOn \"2017-09-22 20:18:19\";\n" \
                   "\tpipeline:hasTag \"arts and entertainment\";\n" \
                   "\tpipeline:hasSourceURL \"https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm\";\n" \
                   "\tpipeline:hasScore 0.632134442843743;\n" \
                   "\tkglids:isPartOf <http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries> .\n"

        pipelines = [
            {
                "url": "https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm",
                "title": "NHL player salary Prediction - XGBoost, rf and SVM",
                "author": "Cam Nugent",
                "votes": 4,
                "score": 0.632134442843743,
                "date": "2017-09-22 20:18:19",
                "tags": ["arts and entertainment"],
                "uri": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm",
                "dataset": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries"
            },
            {
                "url": "https://www.kaggle.com/camnugent/nhl-player-salary-prediction-xgboost-rf-and-svm",
                "title": "NHL player salary Prediction - XGBoost, rf and SVM",
                "author": "Cam Nugent",
                "votes": 4,
                "score": 0.632134442843743,
                "date": "2017-09-22 20:18:19",
                "tags": ["arts and entertainment"],
                "uri": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries/camnugent.nhl-player-salary-prediction-xgboost-rf-and-svm",
                "dataset": "http://kglids.org/resource/kaggle/camnugent.predict-nhl-player-salaries"
            }
        ]

        result = build_default_rdf_page(pipelines)

        self.assertEqual(expected, result)

    def test_title_to_rdf(self):
        expected = f'\trdfs:label "\'I sneezed and now I can\'t move\' Back pain study";\n'
        title = '"I sneezed and now I can\'t move" Back pain study'

        result = title_to_rdf(title)

        print(result)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
