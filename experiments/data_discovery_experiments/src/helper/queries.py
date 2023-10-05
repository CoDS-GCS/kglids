import io
import stardog
import pandas as pd


def get_prefixes():
    return """
    PREFIX kglids:  <http://kglids.org/ontology/>
    PREFIX data:    <http://kglids.org/ontology/data/>
    PREFIX schema:  <http://schema.org/>
    PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
    """


def get_all_profiled_tables():
    return get_prefixes() + """
    SELECT ?table_name
    WHERE
    {
        ?table_id rdf:type      kglids:Table    .
        ?table_id schema:name   ?table_name     .
    }
    """


def get_top_k_tables(pairs: list):
    top_k = {}
    for p in pairs:
        if p[0] not in top_k:
            top_k[p[0]] = p[1]
        else:
            updated_score = top_k.get(p[0]) + p[1]
            top_k[p[0]] = updated_score

    top_k = list(dict(sorted(top_k.items(), key=lambda item: item[1], reverse=True)).keys())
    top_k = [tuple(ele) for ele in top_k]
    return top_k


def get_similar_relation_tables_query(query_table: str, thresh: float):
    return get_prefixes() + """
    SELECT ?table_name1 ?table_name2 ?certainty
    WHERE
    {
        ?table_id	schema:name		"%s"	;
      				schema:name		?table_name1	.
      	?column_id	kglids:isPartOf ?table_id		.

      	<<?column_id data:hasLabelSimilarity	?column_id2>>	data:withCertainty	?certainty	. 

      	FILTER (?certainty >= %s)					.
      	?column_id2 kglids:isPartOf	?table_id2		.
      	?table_id2	schema:name		?table_name2	.
    }
    """ % (query_table, thresh)


def get_related_columns_between_2_tables_query(table_name1, table_name2, relationship: str, thresh: float):
    return get_prefixes() + \
           'SELECT DISTINCT ?table_id1 ?column_name1 ?table_id2 ?column_name2 \nWHERE\n{\n' \
           '    ?table_id1   	schema:name					"%s"                                        .\n' \
           '    ?table_id2   	schema:name					"%s"                                        .\n' \
           '    ?column_id1		kglids:isPartOf				?table_id1   		                                .\n' \
           '    ?column_id2		kglids:isPartOf				?table_id2   		                                .\n' \
           '    <<?column_id1        data:%s            ?column_id2>> data:withCertainty ?c                        .\n' \
           '    FILTER(?c >= %s)                                                                        .\n' \
           '    ?column_id1        schema:name             ?column_name1                                           .\n' \
           '    ?column_id2        schema:name             ?column_name2                                           .\n' \
           '}' % (table_name1, table_name2, relationship, thresh)


def attribute_precision_j_query(query_table, table, thresh: float):
    return get_prefixes() + """

    SELECT DISTINCT  ?target_table ?target_attribute ?joinable_tables_name ?candidate_attribute ?certainty
    WHERE 
    {	
        {
            SELECT DISTINCT ?joinable_tables_name 
            WHERE
                {
                    {
                        ?table_id1	schema:name		"%s"			. #si
                        ?column1	kglids:isPartOf	?table_id1				.
                        ?column1 	data:hasContentSimilarity		?column2				.	
                        ?column2	kglids:isPartOf	?table_id2				.
                        ?table_id2	rdf:type			kglids:Table			.
                        ?table_id2	schema:name		?joinable_tables_name	.
                    }

                    UNION                               # 1 hop
                    {
                        ?table_id3	schema:name		?joinable_tables_name	.
                        ?column3	kglids:isPartOf	?table_id3				.
                        ?column3	data:hasContentSimilarity		?column4				.
                        ?column4	kglids:isPartOf	?table_id4				.
                        ?table_id4	rdf:type			kglids:Table			.
                        ?table_id4	schema:name		?joinable_tables_name	.
                    }

                        UNION                           # 2 hop
                    {
                        ?table_id5	schema:name		?joinable_tables_name	.
                        ?column5	kglids:isPartOf	?table_id5				.
                        ?column5	data:hasContentSimilarity		?column6				.
                        ?column6	kglids:isPartOf	?table_id6				.
                        ?table_id6	rdf:type			kglids:Table			.
                        ?table_id6	schema:name		?joinable_tables_name	.
                    }

                        UNION                           # 3 hop
                    {
                        ?table_id7	schema:name		?joinable_tables_name	.
                        ?column7	kglids:isPartOf	?table_id7				.
                        ?column7	data:hasContentSimilarity		?column8				.
                        ?column8	kglids:isPartOf	?table_id8				.
                        ?table_id8	rdf:type			kglids:Table			.
                        ?table_id8	schema:name		?joinable_tables_name	.
                    }

                        UNION                              # 4 hop
                    {
                        ?table_id9	schema:name		?joinable_tables_name	.
                        ?column9	kglids:isPartOf	?table_id9				.
                        ?column9	data:hasContentSimilarity		?column10				.
                        ?column10	kglids:isPartOf	?table_id10				.
                        ?table_id10	rdf:type			kglids:Table			.
                        ?table_id10	schema:name		?joinable_tables_name	.
                    }

                        UNION                              # 5 hop
                    {
                        ?table_id11	schema:name		?joinable_tables_name	.
                        ?column11	kglids:isPartOf	?table_id11				.
                        ?column11	data:hasContentSimilarity		?column12				.
                        ?column12	kglids:isPartOf	?table_id12				.
                        ?table_id12	rdf:type			kglids:Table			.
                        ?table_id12	schema:name		?joinable_tables_name	.
                    }    
                }
        }	
    ?table_id_x		schema:name		?joinable_tables_name	.
    ?table_id_t		schema:name		"%s"			. #Target
    ?table_id_t		schema:name		?target_table			.
    ?column_x		kglids:isPartOf	?table_id_x				.
    ?column_t		kglids:isPartOf	?table_id_t				.
    ?column_t       schema:name     ?target_attribute       .
    ?column_x		schema:name		?candidate_attribute	.
    <<?column_t		data:hasLabelSimilarity		?column_x>>	data:withCertainty	?certainty.		
    FILTER(?certainty >= %s)  .                                                            

    }""" % (table, query_table, thresh)


# --------------------QUERY EXEC-------------------------

def execute_query(conn: stardog.Connection, query: str, return_type: str = 'json', timeout: int = 0):
    if return_type == 'csv':
        result = conn.select(query, content_type='text/csv', timeout=timeout)
        return pd.read_csv(io.BytesIO(bytes(result)))
    elif return_type == 'json':
        result = conn.select(query)
        return result['results']['bindings']
    elif return_type == 'ask':
        result = conn.select(query)
        return result['boolean']
    elif return_type == 'update':
        result = conn.update(query)
        return result
    else:
        error = return_type + ' not supported!'
        raise ValueError(error)


attr_pairs = {}


def get_related_columns_between_2_tables_attribute_precision(sparql, table1, table2, thresh):
    if (table1, table2) in attr_pairs:
        return attr_pairs.get((table1, table2))

    else:
        result = []
        res = execute_query(sparql,
                            get_related_columns_between_2_tables_query(table1, table2, 'hasLabelSimilarity', thresh))
        for r in res:
            c1 = r["column_name1"]["value"]
            c2 = r["column_name2"]["value"]
            result.append((table1, c1, table2, c2))

        attr_pairs[(table1, table2)] = result
        return attr_pairs.get((table1, table2))


attr_pairs_j = {}


def get_related_columns_between_2_tables_j_attribute_precision(SPARQL, query_table: str, table: str, thresh):
    if (query_table, table) in attr_pairs_j:
        return attr_pairs_j.get((query_table, table))
    else:
        result = []
        res = execute_query(SPARQL, attribute_precision_j_query(query_table, table, thresh))

        for r in res:
            target_t = r["target_table"]["value"]
            target_attr = r["target_attribute"]["value"]
            candidate_t = r["joinable_tables_name"]["value"]
            candidate_attr = r["candidate_attribute"]["value"]
            result.append((target_t, target_attr, candidate_t, candidate_attr))

        attr_pairs_j[(query_table, table)] = result
        return attr_pairs_j.get((query_table, table))


top_k_per_target_table = {}


def get_top_k_related_tables_with_cache(sparql, query_table, k, thresh):
    if query_table in top_k_per_target_table:
        top_k = top_k_per_target_table.get(query_table)
        return top_k[:k]

    else:
        result = []
        res = execute_query(sparql, get_similar_relation_tables_query(query_table, thresh))
        for r in res:
            table1 = r["table_name1"]["value"]
            table2 = r["table_name2"]["value"]
            certainty = float(r["certainty"]["value"])
            result.append([(table1, table2), certainty])
        top_k_per_target_table[query_table] = get_top_k_tables(result)
        return top_k_per_target_table.get(query_table)[:k]


def get_top_k_related_tables(sparql, query_table, k, thresh):

    result = []
    res = execute_query(sparql, get_similar_relation_tables_query(query_table, thresh))
    for r in res:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        certainty = float(r["certainty"]["value"])
        result.append([(table1, table2), certainty])
    result = get_top_k_tables(result)
    return result[:k]
