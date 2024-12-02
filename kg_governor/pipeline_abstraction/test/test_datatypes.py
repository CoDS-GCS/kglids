import unittest

import kg_governor.pipeline_abstraction.util as util
from kg_governor.pipeline_abstraction.Calls import CallType
from kg_governor.pipeline_abstraction.datatypes import GraphInformation

PYTHON_FILE = 'test.py'
SOURCE = 'kaggle'
DATASET_NAME = 'titanic'


class TestGraphInformation(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphInformation(PYTHON_FILE, SOURCE, DATASET_NAME)
        self.graph.add_node('')

    def test_add_import_library(self):
        self.graph.add_import_node('pandas')

        self.assertIn('http://kglids.org/resource/library/pandas', self.graph.libraries.keys())
        self.assertEqual('http://kglids.org/resource/library/pandas',
                         self.graph.libraries.get('http://kglids.org/resource/library/pandas').uri)
        self.assertEqual('http://kglids.org/ontology/Library',
                         self.graph.libraries.get('http://kglids.org/resource/library/pandas').type)

    def test_add_import_package(self):
        self.graph.add_import_node('sklearn.ensemble')

        self.assertIn('http://kglids.org/resource/library/sklearn', self.graph.libraries.keys())
        library = self.graph.libraries.get('http://kglids.org/resource/library/sklearn')
        self.assertEqual('http://kglids.org/ontology/Library', library.type)
        self.assertIn('http://kglids.org/resource/library/sklearn/ensemble', library.contain.keys())
        self.assertEqual('http://kglids.org/ontology/Package',
                         library.contain.get('http://kglids.org/resource/library/sklearn/ensemble').type)

    def test_add_import_class(self):
        self.graph.add_import_node('pandas.DataFrame')

        self.assertIn('http://kglids.org/resource/library/pandas', self.graph.libraries.keys())
        library = self.graph.libraries.get('http://kglids.org/resource/library/pandas')
        self.assertEqual('http://kglids.org/ontology/Library', library.type)
        self.assertIn('http://kglids.org/resource/library/pandas/DataFrame', library.contain.keys())
        self.assertEqual('http://kglids.org/ontology/Class',
                         library.contain.get('http://kglids.org/resource/library/pandas/DataFrame').type)

    def test_add_import_function(self):
        self.graph.add_import_node('pandas.DataFrame.drop')

        self.assertIn('http://kglids.org/resource/library/pandas', self.graph.libraries.keys())
        library = self.graph.libraries.get('http://kglids.org/resource/library/pandas')
        self.assertEqual('http://kglids.org/ontology/Library', library.type)
        self.assertIn('http://kglids.org/resource/library/pandas/DataFrame', library.contain.keys())
        a_class = library.contain.get('http://kglids.org/resource/library/pandas/DataFrame')
        self.assertEqual('http://kglids.org/ontology/Class', a_class.type)
        self.assertIn('http://kglids.org/resource/library/pandas/DataFrame/drop', a_class.contain.keys())
        self.assertEqual('http://kglids.org/ontology/Function',
                         a_class.contain.get('http://kglids.org/resource/library/pandas/DataFrame/drop').type)

    def test_add_import_without_significance_return_none(self):
        self.graph.add_built_in_call('len')

        self.assertEqual(None, self.graph.libraries.get('http://kglids.org/resource/library/builtin/len').type)

    def test_add_import_when_new_library_then_add_call_type(self):
        library_name = 'pandas.DataFrame'
        self.graph.add_import_node(library_name)

        pd_path = util.create_import_uri('pandas')
        df_path = util.create_import_uri('pandas.DataFrame')

        self.assertIn(pd_path, self.graph.libraries.keys())
        self.assertEqual(CallType.LIBRARY.value, self.graph.libraries.get(pd_path).type)
        pd = self.graph.libraries.get(pd_path)
        self.assertIn(df_path, pd.contain.keys())
        self.assertEqual(CallType.CLASS.value, pd.contain.get(df_path).type)


class TestLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphInformation(PYTHON_FILE, SOURCE, DATASET_NAME)
        self.graph.add_node('')

    def test_str_when_called_then_return_json_information_about_library(self):
        library_name = 'pandas.DataFrame'
        self.graph.add_import_node(library_name)

        libs = [library.str() for library in self.graph.libraries.values()]

        pd = libs[0]
        self.assertEqual(CallType.LIBRARY.value, pd.get('type'))
        df = pd.get('contain')[0]
        self.assertEqual(CallType.CLASS.value, df.get('type'))



if __name__ == '__main__':
    unittest.main()