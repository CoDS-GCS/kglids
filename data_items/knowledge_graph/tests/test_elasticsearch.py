import sys

sys.path.insert(0, '../src')

from storage.elasticsearch_client import ElasticsearchClient

es_client = ElasticsearchClient()


def test_get_all_fields_num_signatures():
    num_signatures = sorted(es_client.get_num_stats())
    assert (len(num_signatures) == 110)
    assert (num_signatures[0] == ('100886828', (5267.0, 9078.0, 1.0, 11697.0)))


def test_get_all_mh_text_signatures():
    minhash_signatures = sorted(es_client.get_profiles_minhash())
    assert (len(minhash_signatures) == 179)
    assert (minhash_signatures[0][0] == '1013079926')
    assert (len(minhash_signatures[0][1]) == 512)


def test_get_all_fields():
    fields = sorted(es_client.get_profile_attributes())
    assert (len(fields) == 289)
    print(fields[0])
    assert (fields[0] == ('100886828', '', 'countries', 'target_components.csv', 'targcomp_id', 9052, 9052, 'N')
            )
