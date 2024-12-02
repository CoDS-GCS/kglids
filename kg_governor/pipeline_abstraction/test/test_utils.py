import unittest
from kg_governor.pipeline_abstraction.Calls import pd_dataframe
import kg_governor.pipeline_abstraction.util as util


class TestUtilsFunctions(unittest.TestCase):
    def test_get_package_when_Calls_package_return_package(self):
        package = pd_dataframe
        result = util.get_package(package, '')

        self.assertEqual(package.name, result.name)

    def test_get_package_when_None_return_None(self):
        result = util.get_package(None, '')

        self.assertIsNone(result)

    def test_get_package_when_name_not_string_return_None(self):
        name = dict()
        result = util.get_package(name, '')

        self.assertIsNone(result)

    def test_get_package_when_sub_package_in_name_and_parent_in_alias_return_right_call_package(self):
        name = 'pd.DataFrame'
        alias = {
            'pd': 'pandas'
        }
        expected_result = pd_dataframe

        result = util.get_package(name, alias)

        self.assertEqual(expected_result.name, result.name)

    def test_get_package_when_sub_package_in_name_and_parent_not_in_alias_return_right_call_package(self):
        name = 'pandas.DataFrame'
        expected_result = pd_dataframe

        result = util.get_package(name, {})

        self.assertEqual(expected_result.name, result.name)


if __name__ == '__main__':
    unittest.main()
