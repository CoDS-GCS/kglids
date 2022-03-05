import config as c


def get_search_template(var, keyword, limit):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>\n' \
           'prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
           'prefix dct: <http://purl.org/dc/terms/>\n' \
           'select ?id ?dn ?fn ?cn ?dt\n' \
           'where {values %s {"%s"}\n' \
           '       ?id schema:name ?cn.\n' \
           '       ?id a lac:column.\n' \
           '       ?id schema:type ?dt.\n' \
           '       ?id dct:isPartOf ?fid.\n' \
           '       ?fid a lac:table.\n' \
           '       ?fid schema:name ?fn.\n' \
           '       ?fid dct:isPartOf ?did.\n' \
           '       ?did a lac:dataset.\n' \
           '       ?did schema:name ?dn.}' \
           'limit %s' % (var, keyword, limit)


def get_approximate_search_template(var, keyword, limit):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>\n' \
           'prefix dct: <http://purl.org/dc/terms/>\n' \
           'prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
           'select ?id ?dn ?fn ?cn ?dt\n' \
           'where {?id schema:name ?cn.\n' \
           '       ?id a lac:column.\n' \
           '       ?id schema:type ?dt.\n' \
           '       ?id dct:isPartOf ?fid.\n' \
           '       ?fid a lac:table.' \
           '       ?fid schema:name ?fn.\n' \
           '       ?fid dct:isPartOf ?did.\n' \
           '       ?did a lac:dataset.\n' \
           '       ?did schema:name ?dn.\n' \
           'filter regex(%s , "%s", "i").}' \
           'limit %s' % (var, keyword.replace('_', '|'), limit)


def get_related_neighbors_template(relation, classIDs, limit):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>\n' \
           'prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
           'prefix dct: <http://purl.org/dc/terms/>\n' \
           'select ?id ?cn ?fn ?dn ?score ?dt where{\n' \
           '<<?is lac:%s ?id>> lac:certainty ?score.\n' \
           '?is a lac:column.\n' \
           '?id a lac:column.\n' \
           '?id schema:type ?dt.\n' \
           '?id dct:isPartOf ?fid.\n' \
           '?fid a lac:table.\n' \
           '?fid schema:name ?fn.\n' \
           '?id schema:name ?cn.\n' \
           '?fid dct:isPartOf ?did.\n' \
           '?did a lac:dataset.\n' \
           '?did schema:name ?dn.\n' \
           'values ?is {%s}.}' \
           'limit %s' % (relation.name, classIDs, limit)


def get_path_between_nodes_template(aID, bID, relation, maxHops):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>\n' \
           'prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
           'prefix gas: <http://www.bigdata.com/rdf/gas#>\n' \
           'prefix dct: <http://purl.org/dc/terms/>\n' \
           'SELECT ?predecessor ?cn1 ?fn1 ?dn1 ?depth ?score ?out ?cn2 ?fn2 ?dn2 ?dt1 ?dt2{\n' \
           'SERVICE gas:service {\n' \
           'gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP".\n' \
           'gas:program gas:in %s.\n' \
           'gas:program gas:target %s.\n' \
           'gas:program gas:out ?out.\n' \
           'gas:program gas:out1 ?depth.\n' \
           'gas:program gas:out2 ?predecessor.\n' \
           'gas:program gas:linkType lac:%s.\n' \
           'gas:program gas:maxIterations %d .\n' \
           '}\n' \
           'filter (?out != %s).\n' \
           '?predecessor schema:name ?cn1.\n' \
           '?predecessor a lac:column.\n' \
           '?predecessor schema:type ?dt1.\n' \
           '?predecessor dct:isPartOf ?fid1.\n' \
           '?fid1 a lac:table.\n' \
           '?fid1 schema:name ?fn1.\n' \
           '?fid1 dct:isPartOf ?did1.\n' \
           '?did1 a lac:dataset.\n' \
           '?did1 schema:name ?dn1.\n' \
           '?out schema:name ?cn2.\n' \
           '?out a lac:column.\n' \
           '?out schema:type ?dt2.\n' \
           '?out dct:isPartOf ?fid2.\n' \
           '?fid2 a lac:table.\n' \
           '?fid2 schema:name ?fn2.\n' \
           '?fid2 dct:isPartOf ?did2.\n' \
           '?did2 a lac:dataset.\n' \
           '?did2 schema:name ?dn2.\n' \
           '<<?predecessor lac:%s ?out>> lac:certainty ?score.\n' \
           '}\n' \
           'order by (?depth)' % (aID, bID, relation.name, maxHops, aID, relation.name)


def get_number_of_datasets():
    return c.prefix + \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           'select (count(?cid) as ?count)' \
           ' where {?cid a lac:dataset}'


def get_tables_in_dataset():
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>\n' \
           'prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
           'prefix dct: <http://purl.org/dc/terms/>\n' \
           'select ?dn ?cn where{\n' \
           'values ?dn {"%s"}\n' \
           '?cid dct:isPartOf ?did.\n' \
           '?cid schema:name ?cn.\n' \
           '?did schema:name ?dn.\n' \
           '?did a lac:dataset.}\n' \
           'order by ?cn\n' \
           'limit %s'


def get_all_tables():
    return c.prefix + \
           '\nprefix dct: <http://purl.org/dc/terms/>\n' \
           'prefix schema: <http://schema.org/>\n' \
           'select ?dn ?cn where{\n' \
           '?cid dct:isPartOf ?did.\n' \
           '?did a lac:dataset.\n' \
           '?cid schema:name ?cn.\n' \
           '?did schema:name ?dn.}\n' \
           'order by ?cn'


# get columns with label containing a keyword
def search_columns(keyword: str) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix dct: <http://purl.org/dc/terms/>' \
           '\nprefix owl: <http://www.w3.org/2002/07/owl#>' \
           '\nselect ?total ?name ?distinct ?missing ?type ?origin ?cardinality ?min ?max ?median ?table_name ' \
           '?dataset_name' \
           '\nwhere {?column rdfs:label ?label.' \
           '\n?column rdf:type lac:column.' \
           '\n?column schema:totalVCount ?total.' \
           '\n?column schema:name ?name.' \
           '\n?column schema:distinctVCount ?distinct.' \
           '\n?column schema:missingVCount ?missing.' \
           '\n?column schema:type ?type.' \
           '\n?column lac:origin ?origin.' \
           '\n?column owl:cardinality ?cardinality.' \
           '\n?column dct:isPartOf ?table.' \
           '\n?table schema:name ?table_name.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\noptional {' \
           '\n?column schema:minValue ?min.' \
           '\n?column schema:maxValue ?max.' \
           '\n?column schema:median ?median.' \
           '\n}' \
           '\nfilter( regex(?label, "' + keyword + '", "i" ))}'


# get the table with the label containing a keyword
def search_tables(keyword: str) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>prefix' \
           '\n dct: <http://purl.org/dc/terms/>' \
           '\nselect ?name ?dataset_name ?origin ?path (count(*) as ?number_of_columns) (max(?total) as ' \
           '?number_of_rows)' \
           '\nwhere {' \
           '\n?table rdfs:label ?label.' \
           '\n?table rdf:type lac:table.' \
           '\n?table lac:path ?path.' \
           '\n?table schema:name ?name.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\n?column dct:isPartOf ?table.' \
           '\n?column rdf:type lac:column.' \
           '\n?column lac:origin ?origin.' \
           '\n?column schema:totalVCount ?total' \
           '\nfilter( regex(?label, "' + keyword + '", "i" ))}' \
                                                   '\ngroup by ?name ?dataset_name ?origin ?path'


def search_table_by_name(keyword: str) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>prefix' \
           '\n dct: <http://purl.org/dc/terms/>' \
           '\nselect ?name ?dataset_name ?origin ?path (count(*) as ?number_of_columns) (max(?total) as ' \
           '?number_of_rows)' \
           '\nwhere {' \
           '\n?table rdfs:label ?label.' \
           '\n?table rdf:type lac:table.' \
           '\n?table lac:path ?path.' \
           '\n?table schema:name ?name.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\n?column dct:isPartOf ?table.' \
           '\n?column rdf:type lac:column.' \
           '\n?column lac:origin ?origin.' \
           '\n?column schema:totalVCount ?total.' \
           '\nvalues ?name {\"' + keyword + '\"}}' \
           '\ngroup by ?name ?dataset_name ?origin ?path'


def search_tables_on(conditions: tuple):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix dct: <http://purl.org/dc/terms/>' \
           '\nselect ?name ?dataset_name ?origin ?path (' \
           'count(distinct ?cols) as ?number_of_columns) (max (?total) as ?number_of_rows)' \
           '\nwhere {' \
           '\n?table schema:name ?name.' \
           '\n?table lac:path ?path.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\n?cols dct:isPartOf ?table.' \
           '\n?cols schema:totalVCount ?total.\n' \
           + conditions[0] + \
           '\nfilter( ' + conditions[1] + ')}' \
                                          '\n group by ?name ?dataset_name ?origin ?path'


def get_joinable_columns(ids: list):
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix dct: <http://purl.org/dc/terms/>' \
           '\nprefix owl: <http://www.w3.org/2002/07/owl#>' \
           '\nselect ?total ?name ?distinct ?missing ?type ?origin ?cardinality ?min ?max ?median ?table_name ' \
           '?dataset_name' \
           '\nwhere {?column rdfs:label ?label.' \
           '\n?column rdf:type lac:column.' \
           '\n?column schema:totalVCount ?total.' \
           '\n?column schema:name ?name.' \
           '\n?column schema:distinctVCount ?distinct.' \
           '\n?column schema:missingVCount ?missing.' \
           '\n?column schema:type ?type.' \
           '\n?column lac:origin ?origin.' \
           '\n?column owl:cardinality ?cardinality.' \
           '\n?column dct:isPartOf ?table.' \
           '\n?table schema:name ?table_name.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\noptional {' \
           '\n?column schema:minValue ?min.' \
           '\n?column schema:maxValue ?max.' \
           '\n?column schema:median ?median.' \
           '\n}' \
           'values ?column {%s}}' % (ids)


def get_shortest_path_between_columns(in_id: int, target_id: int, via: str, max_hops: int) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix gas: <http://www.bigdata.com/rdf/gas#>' \
           '\nprefix owl: <http://www.w3.org/2002/07/owl#>' \
           '\nprefix dct: <http://purl.org/dc/terms/>' \
           '\nSELECT ?name ?table_name ?dataset_name ?total ?distinct ?missing ?origin ?cardinality ?type ?min ?max ?median {' \
           '\nSERVICE gas:service {' \
           '      \ngas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP".' \
           '      \ngas:program gas:in lac:%s.' \
           '      \ngas:program gas:target lac:%s.' \
           '      \ngas:program gas:out ?out.' \
           '      \ngas:program gas:out1 ?depth.' \
           '      \ngas:program gas:out2 ?predecessor.' \
           '      \ngas:program gas:linkType lac:%s.' \
           '      \ngas:program gas:maxIterations %s .' \
           '      \n}' \
           '\n?out a lac:column.' \
           '\n?out schema:name ?name.' \
           '\n?out dct:isPartOf ?table.' \
           '\n?table schema:name ?table_name.' \
           '\n?table dct:isPartOf ?dataset.' \
           '\n?dataset schema:name ?dataset_name.' \
           '\n?out schema:totalVCount ?total.' \
           '\n?out schema:distinctVCount ?distinct.' \
           '\n?out schema:missingVCount ?missing.' \
           '\n?out lac:origin ?origin.' \
           '\n?out owl:cardinality ?cardinality.' \
           '\n?out schema:type ?type.' \
           '\noptional {' \
           '      \n?out schema:minValue ?min.' \
           '      \n?out schema:maxValue ?max.' \
           '      \n?out schema:median ?median.' \
           '      \n}' \
           '\n}' \
           '\norder by (?depth)' % (in_id, target_id, via, max_hops)


def get_iri_of_dataset(name: str) -> str:
    return c.prefix + \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix schema: <http://schema.org/>' \
           '\nselect ?id' \
           '\nwhere {' \
           '\n?id a lac:dataset.' \
           '\n?id rdfs:label %s }' % name


def get_iri_of_table(dataset_name: str, table_name: str) -> str:
    return c.prefix + \
           '\nprefix dct: <http://purl.org/dc/terms/>' \
           '\nprefix schema: <http://schema.org/>' \
           '\nselect ?id where{' \
           '\n?id a lac:table.' \
           '\n?id rdfs:label %s.' \
           '\n?id dct:isPartOf ?dataset.' \
           '\n?dataset rdfs:label %s.' \
           '\n?dataset a lac:dataset.}' % (table_name, dataset_name)


def get_num_of_annotations_in(iri: str, predicate: str) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nselect (count(?annotations) as ?num_annotation)' \
           '\nwhere{' \
           '\n%s %s ?blank.' \
           '\n?blank ?p ?annotations}' % (iri, predicate)


def add_first_element(iri: str, annotation: str, predicate: str) -> str:
    return c.prefix + \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix schema: <http://schema.org/>' \
           '\ninsert data {' \
           '\n%s %s [a rdf:Bag; rdf:_1 %s].}' % (iri, predicate, annotation)


def add_element(iri: str, annotation: str, predicate: str, num_used_in: int) -> str:
    return c.prefix + \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix schema: <http://schema.org/>' \
           '\nINSERT {' \
           '\n?blank rdf:_%s %s.}' \
           '\nWHERE {' \
           '\n%s %s ?blank .}' % (num_used_in, annotation, iri, predicate)


def get_usages(iri: str, predicate: str) -> str:
    return c.prefix + \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix schema: <http://schema.org/>' \
           '\nselect ?annotation where {' \
           '\n%s %s ?blank.' \
           '\n ?blank ?p ?annotation.' \
           '\nfilter (?p != rdf:type)}' % (iri, predicate)


def get_paths_between_tables(select: str, start_iri: str, nodes: str, relationships: str, target_iri: str,
                             hops: int) -> str:
    return c.prefix + \
           '\nprefix schema: <http://schema.org/>' \
           '\nprefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>' \
           '\nprefix owl: <http://www.w3.org/2002/07/owl#>' \
           '\nprefix dct: <http://purl.org/dc/terms/>' + \
           '\n select' + select + \
           '\nwhere {' + \
           nodes + relationships + \
           '\n values ?t1 {' + start_iri + '}' + \
           '\n values ?t' + str(hops + 1) + ' {' + target_iri + '}}'
