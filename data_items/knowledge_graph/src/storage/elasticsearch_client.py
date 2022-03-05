import json

import config as c
from elasticsearch import Elasticsearch


class ElasticsearchClient:
    # Store client
    client = None

    def __init__(self):
        """
            Uses the configuration file to create a connection to the store
            :return:
            """
        global client
        client = Elasticsearch([{'host': c.db_host, 'port': c.db_port}], timeout=30)

    def get_profile_attributes(self):
        """
        Reads all fields, described as (id, source_name, field_name) from the store.
        :return: a list of all fields with the form (id, source_name, field_name)
        """
        body = {"query": {"match_all": {}}}
        res = client.search(index='profiles', body=body, scroll="10m",
                            filter_path=['_scroll_id',
                                         'hits.total',
                                         'hits.hits._source.id',
                                         'hits.hits._source.origin',
                                         'hits.hits._source.datasetName',
                                         'hits.hits._source.datasetid',
                                         'hits.hits._source.tableName',
                                         'hits.hits._source.tableid',
                                         'hits.hits._source.datasource',
                                         'hits.hits._source.columnName',
                                         'hits.hits._source.totalValuesCount',
                                         'hits.hits._source.distinctValuesCount',
                                         'hits.hits._source.missingValuesCount',
                                         'hits.hits._source.dataType',
                                         'hits.hits._source.median',
                                         'hits.hits._source.minValue',
                                         'hits.hits._source.maxValue',
                                         'hits.hits._source.path'
                                         ]
                            )
        scroll_id = res['_scroll_id']
        remaining = res['hits']['total']['value']
        while remaining > 0:
            hits = res['hits']['hits']
            for h in hits:
                id_source_and_file_name = (str(h['_source']['id']),
                                           h['_source']['origin'],
                                           h['_source']['datasetName'],
                                           h['_source']['datasetid'],
                                           h['_source']['tableName'],
                                           h['_source']['tableid'],
                                           h['_source']['datasource'],
                                           h['_source']['columnName'],
                                           int(h['_source']['totalValuesCount']),
                                           int(h['_source']['distinctValuesCount']),
                                           int(h['_source']['missingValuesCount']),
                                           h['_source']['dataType'],
                                           int(h['_source']['median']),
                                           int(h['_source']['minValue']),
                                           int(h['_source']['maxValue']),
                                           h['_source']['path'])
                yield id_source_and_file_name
                remaining -= 1
            res = client.scroll(scroll="5m", scroll_id=scroll_id,
                                filter_path=['_scroll_id',
                                             'hits.hits._source.id',
                                             'hits.hits._source.origin',
                                             'hits.hits._source.datasetName',
                                             'hits.hits._source.datasetid',
                                             'hits.hits._source.tableName',
                                             'hits.hits._source.tableid',
                                             'hits.hits._source.datasource',
                                             'hits.hits._source.columnName',
                                             'hits.hits._source.totalValuesCount',
                                             'hits.hits._source.distinctValuesCount',
                                             'hits.hits._source.missingValuesCount',
                                             'hits.hits._source.dataType',
                                             'hits.hits._source.median',
                                             'hits.hits._source.minValue',
                                             'hits.hits._source.maxValue',
                                             'hits.hits._source.path'
                                             ]
                                )
            scroll_id = res['_scroll_id']  # update the scroll_id
        client.clear_scroll(scroll_id=scroll_id)

    def get_profiles_minhash(self, string_type):
        """
        Retrieves id-mh fields
        :return: (fields, numsignatures)
        """
        query_body = {"query": {"match": {"dataType": string_type}}}
        res = client.search(index='profiles', body=query_body, scroll="10m",
                            filter_path=['_scroll_id',
                                         'hits.total',
                                         'hits.hits._source.id',
                                         'hits.hits._source.minhash']
                            )
        scroll_id = res['_scroll_id']
        remaining = res['hits']['total']['value']

        id_sig = []
        while remaining > 0:
            hits = res['hits']['hits']
            for h in hits:
                data = (str(h['_source']['id']), json.loads(h['_source']['minhash']))
                id_sig.append(data)
                remaining -= 1
            res = client.scroll(scroll="5m", scroll_id=scroll_id,
                                filter_path=['_scroll_id',
                                             'hits.hits._source.id',
                                             'hits.hits._source.minhash']
                                )
            scroll_id = res['_scroll_id']  # update the scroll_id
        client.clear_scroll(scroll_id=scroll_id)
        return id_sig

    def get_profiles_deep_embeddings(self):
        """
        Retrieves id_de fields
        :return: (fields, numsignatures_de)
        """
        query_body = {"query": {"match": {"dataType": "N"}}}
        res = client.search(index='profiles', body=query_body, scroll="10m",
                            filter_path=['_scroll_id',
                                         'hits.total',
                                         'hits.hits._source.id',
                                         'hits.hits._source.deep_embeddings']
                            )
        scroll_id = res['_scroll_id']
        remaining = res['hits']['total']['value']

        id_sig_de = []
        while remaining > 0:
            hits = res['hits']['hits']
            for h in hits:
                data = (str(h['_source']['id']), json.loads(h['_source']['deep_embeddings']))
                id_sig_de.append(data)
                remaining -= 1
            res = client.scroll(scroll="5m", scroll_id=scroll_id,
                                filter_path=['_scroll_id',
                                             'hits.hits._source.id',
                                             'hits.hits._source.deep_embeddings']
                                )
            scroll_id = res['_scroll_id']  # update the scroll_id
        client.clear_scroll(scroll_id=scroll_id)
        return id_sig_de


    def get_num_stats(self):
        """
        Retrieves numerical fields and signatures from the store
        :return: (fields, numsignatures)
        """
        query_body = {"query": {"match": {"dataType": "N"}}}
        res = client.search(index='profiles', body=query_body, scroll="10m",
                            filter_path=['_scroll_id',
                                         'hits.total',
                                         'hits.hits._source.id',
                                         'hits.hits._source.median',
                                         'hits.hits._source.iqr',
                                         'hits.hits._source.minValue',
                                         'hits.hits._source.maxValue']
                            )
        scroll_id = res['_scroll_id']
        remaining = res['hits']['total']['value']
        id_sig = []
        while remaining > 0:
            hits = res['hits']['hits']
            for h in hits:
                data = (str(h['_source']['id']), (h['_source']['median'], h['_source']['iqr'],
                                                  h['_source']['minValue'], h['_source']['maxValue']))
                id_sig.append(data)
                remaining -= 1
            res = client.scroll(scroll="5m", scroll_id=scroll_id,
                                filter_path=['_scroll_id',
                                             'hits.hits._source.id',
                                             'hits.hits._source.median',
                                             'hits.hits._source.iqr',
                                             'hits.hits._source.minValue',
                                             'hits.hits._source.maxValue']
                                )
            scroll_id = res['_scroll_id']  # update the scroll_id
        client.clear_scroll(scroll_id=scroll_id)
        return id_sig


if __name__ == "__main__":
    print("Elastic Store")
