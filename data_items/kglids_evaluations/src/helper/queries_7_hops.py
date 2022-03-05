
import operator
from SPARQLWrapper import SPARQLWrapper, JSON

def get_prefixes():
    return """
            PREFIX kglids: <http://kglids.org/>
            PREFIX kglids_data: <http://kglids.org/data/>
            PREFIX schema: <http://schema.org/>
            PREFIX purl:   <http://purl.org/dc/terms/>
            PREFIX w3:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
            """


def check_if_table_exists(query_table: str):
    return get_prefixes() + 'ASK\n{\n?table_id   	schema:name					"%s"\n}' % query_table


def get_all_profiled_tables():
    return get_prefixes() + """
    SELECT ?table_name
    WHERE
    {
        ?table_id w3:type       kglids:table    .
        ?table_id schema:name   ?table_name     .
    }
    """


def get_top_k(query_table: str, k: int):
    return get_prefixes() + \
           'SELECT DISTINCT ?table_name1 ?table_name2 ?certainty\nWHERE\n{\n' \
           '    ?table_id   	schema:name					"%s"                                        ;\n' \
           '                    schema:name					?table_name1    	                                .\n' \
           '    ?column_id		purl:isPartOf				?table_id   		                                .\n' \
           '    <<?column_id	kglids:semanticSimilarity	?column_id2>>   kglids:certainty	?certainty  	.\n' \
           '    ?column_id2		purl:isPartOf				?table_id2  		                                .\n' \
           '    ?table_id2		schema:name					?table_name2    	                                .\n' \
           '}	ORDER BY DESC(?certainty) LIMIT %s' % (query_table, k)


def get_top_k_query(pairs: list, k: int):
    top_k = {}
    secondary_certainty = {}
    for p in pairs:
        p[2] = float(p[2])
        if p[2] <= 0.75: # i.e. ignore values <= threshold
            continue
        if (p[0], p[1]) in top_k:  # if pair exists already
            if p[2] > top_k.get((p[0], p[1])):  # if certainty is greater than what in previous pairs: update
                top_k[(p[0], p[1])] = p[2]
            
            if p[2] <= top_k.get((p[0], p[1])): 
                if (p[0], p[1]) in secondary_certainty:
                    c = secondary_certainty.get((p[0], p[1]))
                    c.append(p[2])
                    secondary_certainty[(p[0], p[1])] = c
                   
                else:
                    secondary_certainty[(p[0], p[1])] = [p[2]]
                    #print("appending for first time: ", secondary_certainty.get((p[0], p[1])))
 
            # top_k[(p[0], p[1])] = p[2] + top_k.get((p[0], p[1]))
                

        else:
            top_k[(p[0], p[1])] = p[2]


    for key in top_k:
        if key in secondary_certainty:
            # print(top_k.get(key))
            # print(sum(secondary_certainty.get(key)))
            # print("\n")
            top_k[key] = top_k.get(key) + sum(secondary_certainty.get(key))

    top_k = dict(sorted(top_k.items(), key=operator.itemgetter(1), reverse=True))

    top_k = list(top_k.keys())
    top_k = [list(ele) for ele in top_k]
    return top_k[:k]


def get_top_k_relation_query(pairs: list, k: int, thresh=0.75):
    top_k = {}
    secondary_certainty = {}
    for p in pairs:
        p[2] = float(p[2])
        if p[2] < thresh:
            continue
        if (p[0], p[1]) in top_k:  # if pair exists already
            if p[2] > top_k.get((p[0], p[1])):  # if certainty is greater than what in previous pairs: update
                top_k[(p[0], p[1])] = p[2]
            
            if p[2] <= top_k.get((p[0], p[1])): 
                if (p[0], p[1]) in secondary_certainty:
                    c = secondary_certainty.get((p[0], p[1]))
                    c.append(p[2])
                    secondary_certainty[(p[0], p[1])] = c
                   
                else:
                    secondary_certainty[(p[0], p[1])] = [p[2]]
                    #print("appending for first time: ", secondary_certainty.get((p[0], p[1])))
 
            # top_k[(p[0], p[1])] = p[2] + top_k.get((p[0], p[1]))
                

        else:
            top_k[(p[0], p[1])] = p[2]


    for key in top_k:
        if key in secondary_certainty:
            # print(top_k.get(key))
            # print(sum(secondary_certainty.get(key)))
            # print("\n")
            top_k[key] = top_k.get(key) + sum(secondary_certainty.get(key))

    top_k = dict(sorted(top_k.items(), key=operator.itemgetter(1), reverse=True))

    top_k = list(top_k.keys())
    top_k = [list(ele) for ele in top_k]
    return top_k[:k]



def get_semantic_similar_tables_query(query_table: str):
    return get_prefixes() + \
           'SELECT ?table_name1 ?table_name2 ?certainty\nWHERE\n{\n' \
           '    ?table_id   	schema:name					"%s"                                        ;\n' \
           '                    schema:name					?table_name1    	                                .\n' \
           '    ?column_id		purl:isPartOf				?table_id   		                                .\n' \
           '    <<?column_id	kglids:semanticSimilarity	?column_id2>>   kglids:certainty	?certainty  	.\n' \
           '    ?column_id2		purl:isPartOf				?table_id2  		                                .\n' \
           '    ?table_id2		schema:name					?table_name2    	                                .\n' \
           '}' % query_table

def get_similar_relation_tables_query(query_table: str, relation: str):
    return get_prefixes() + \
           'SELECT ?table_name1 ?table_name2 ?certainty\nWHERE\n{\n' \
           '    ?table_id   	schema:name					"%s"                                        ;\n' \
           '                    schema:name					?table_name1    	                                .\n' \
           '    ?column_id		purl:isPartOf				?table_id   		                                .\n' \
           '    <<?column_id	kglids:%s               	?column_id2>>   kglids:certainty	?certainty  	.\n' \
           '    ?column_id2		purl:isPartOf				?table_id2  		                                .\n' \
           '    ?table_id2		schema:name					?table_name2    	                                .\n' \
           '}' % (query_table, relation)




def get_all_profiled_tables_query():
    return get_prefixes() + """
    SELECT ?table_name
    WHERE
    {
        ?table_id w3:type       kglids_data:table    .
        ?table_id schema:name   ?table_name     .
    }
    """

def get_all_columns_query(table_name: str):
    return get_prefixes() + \
        'SELECT ?column_id\nWHERE\n{\n' \
        '   ?table_id    schema:name     "%s"       .\n'\
        '   ?column_id   purl:isPartOf   ?table_id   .\n'\
        '}' % table_name

def get_related_columns_between_2_tables_query(table_name1, table_name2, relationship:str):
    return get_prefixes() + \
        'SELECT DISTINCT ?column_id1 ?column_id2 ?column_name1 ?column_name2 ?c\nWHERE\n{\n'\
        '    ?table_id1   	schema:name					"%s"                                        .\n' \
        '    ?table_id2   	schema:name					"%s"                                        .\n' \
        '    ?column_id1		purl:isPartOf				?table_id1   		                                .\n' \
        '    ?column_id2		purl:isPartOf				?table_id2   		                                .\n' \
        '    <<?column_id1        kglids:%s            ?column_id2>> kglids:certainty ?c                        .\n' \
        '    ?column_id1        schema:name             ?column_name1                                           .\n' \
        '    ?column_id2        schema:name             ?column_name2                                           .\n' \
        '}' % (table_name1, table_name2, relationship)

def get_related_columns_between_2_tables_j_query(target_table, si):
    return get_prefixes() + """
    # takes si and T as input
SELECT DISTINCT  ?column_t ?certainty
WHERE 
{	
  	{
    SELECT DISTINCT ?joinable_tables_name 
    WHERE
  	{
    {
        ?table_id1	schema:name		"%s"			. #si
  		?column1	purl:isPartOf	?table_id1				.
	  	?column1 	kglids:pkfk		?column2				.	
  		?column2	purl:isPartOf	?table_id2				.
      	?table_id2	w3:type			kglids_data:table			.
  		?table_id2	schema:name		?joinable_tables_name	.
    }
 
  	UNION                                   # 1 hop
  	{
  		?table_id3	schema:name		?joinable_tables_name	.
      	?column3	purl:isPartOf	?table_id3				.
      	?column3	kglids:pkfk		?column4				.
      	?column4	purl:isPartOf	?table_id4				.
      	?table_id4	w3:type			kglids_data:table			.
      	?table_id4	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 2 hop
  	{
  		?table_id5	schema:name		?joinable_tables_name	.
      	?column5	purl:isPartOf	?table_id5				.
      	?column5	kglids:pkfk		?column6				.
      	?column6	purl:isPartOf	?table_id6				.
      	?table_id6	w3:type			kglids_data:table			.
      	?table_id6	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 3 hop
  	{
  		?table_id7	schema:name		?joinable_tables_name	.
      	?column7	purl:isPartOf	?table_id7				.
      	?column7	kglids:pkfk		?column8				.
      	?column8	purl:isPartOf	?table_id8				.
      	?table_id8	w3:type			kglids_data:table			.
      	?table_id8	schema:name		?joinable_tables_name	.
  	}
	
  	  	UNION                               # 4 hop
  	{
  		?table_id9	schema:name		?joinable_tables_name	.
      	?column9	purl:isPartOf	?table_id9				.
      	?column9	kglids:pkfk		?column10				.
      	?column10	purl:isPartOf	?table_id10				.
      	?table_id10	w3:type			kglids_data:table			.
      	?table_id10	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 5 hop
  	{
  		?table_id11	schema:name		?joinable_tables_name	.
      	?column11	purl:isPartOf	?table_id11				.
      	?column11	kglids:pkfk		?column12				.
      	?column12	purl:isPartOf	?table_id12				.
      	?table_id12	w3:type			kglids_data:table			.
      	?table_id12	schema:name		?joinable_tables_name	.
  	}

      	  	UNION                           # 6 hop
  	{
  		?table_id13	schema:name		?joinable_tables_name	.
      	?column13	purl:isPartOf	?table_id12				.
      	?column13	kglids:pkfk		?column14				.
      	?column14	purl:isPartOf	?table_id14				.
      	?table_id14	w3:type			kglids_data:table			.
      	?table_id14	schema:name		?joinable_tables_name	.
  	}
        	  	UNION                       # 7 hop
  	{
  		?table_id15	schema:name		?joinable_tables_name	.
      	?column15	purl:isPartOf	?table_id15				.
      	?column15	kglids:pkfk		?column16				.
      	?column16	purl:isPartOf	?table_id16				.
      	?table_id16	w3:type			kglids_data:table			.
      	?table_id16	schema:name		?joinable_tables_name	.
  	}
    }
  	}	
  	?table_id_x		schema:name		?joinable_tables_name	.
    ?table_id_t		schema:name		"%s"			. #T
  	?column_x		purl:isPartOf	?table_id_x				.
  	?column_t		purl:isPartOf	?table_id_t				.
    ?column_t       schema:name     ?column_name2           .
  	<<?column_t		kglids:semanticSimilarity		?column_x>>	kglids:certainty	?certainty.		
} 	
""" % (si, target_table)


def attribute_precision_j_query(target_table, si):
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
  		?column1	purl:isPartOf	?table_id1				.
	  	?column1 	kglids:pkfk		?column2				.	
  		?column2	purl:isPartOf	?table_id2				.
      	?table_id2	w3:type			kglids_data:table			.
  		?table_id2	schema:name		?joinable_tables_name	.
    }
 
  	UNION                                   # 1 hop
  	{
  		?table_id3	schema:name		?joinable_tables_name	.
      	?column3	purl:isPartOf	?table_id3				.
      	?column3	kglids:pkfk		?column4				.
      	?column4	purl:isPartOf	?table_id4				.
      	?table_id4	w3:type			kglids_data:table			.
      	?table_id4	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 2 hop
  	{
  		?table_id5	schema:name		?joinable_tables_name	.
      	?column5	purl:isPartOf	?table_id5				.
      	?column5	kglids:pkfk		?column6				.
      	?column6	purl:isPartOf	?table_id6				.
      	?table_id6	w3:type			kglids_data:table			.
      	?table_id6	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 3 hop
  	{
  		?table_id7	schema:name		?joinable_tables_name	.
      	?column7	purl:isPartOf	?table_id7				.
      	?column7	kglids:pkfk		?column8				.
      	?column8	purl:isPartOf	?table_id8				.
      	?table_id8	w3:type			kglids_data:table			.
      	?table_id8	schema:name		?joinable_tables_name	.
  	}
	
  	  	UNION                               # 4 hop
  	{
  		?table_id9	schema:name		?joinable_tables_name	.
      	?column9	purl:isPartOf	?table_id9				.
      	?column9	kglids:pkfk		?column10				.
      	?column10	purl:isPartOf	?table_id10				.
      	?table_id10	w3:type			kglids_data:table			.
      	?table_id10	schema:name		?joinable_tables_name	.
  	}
  
  	  	UNION                               # 5 hop
  	{
  		?table_id11	schema:name		?joinable_tables_name	.
      	?column11	purl:isPartOf	?table_id11				.
      	?column11	kglids:pkfk		?column12				.
      	?column12	purl:isPartOf	?table_id12				.
      	?table_id12	w3:type			kglids_data:table			.
      	?table_id12	schema:name		?joinable_tables_name	.
  	}

      	  	UNION                           # 6 hop
  	{
  		?table_id13	schema:name		?joinable_tables_name	.
      	?column13	purl:isPartOf	?table_id12				.
      	?column13	kglids:pkfk		?column14				.
      	?column14	purl:isPartOf	?table_id14				.
      	?table_id14	w3:type			kglids_data:table			.
      	?table_id14	schema:name		?joinable_tables_name	.
  	}
        	  	UNION                       # 7 hop
  	{
  		?table_id15	schema:name		?joinable_tables_name	.
      	?column15	purl:isPartOf	?table_id15				.
      	?column15	kglids:pkfk		?column16				.
      	?column16	purl:isPartOf	?table_id16				.
      	?table_id16	w3:type			kglids_data:table			.
      	?table_id16	schema:name		?joinable_tables_name	.
  	}     
	}
  	}	
  	?table_id_x		schema:name		?joinable_tables_name	.
    ?table_id_t		schema:name		"%s"			. #Target
  	?table_id_t		schema:name		?target_table			.
  	?column_x		purl:isPartOf	?table_id_x				.
  	?column_t		purl:isPartOf	?table_id_t				.
    ?column_t       schema:name     ?target_attribute       .
  	?column_x		schema:name		?candidate_attribute	.
  	<<?column_t		kglids:semanticSimilarity		?column_x>>	kglids:certainty	?certainty.		
} 	
""" % (si, target_table)

##############QUERY EXEC#####################    

def execute_query(sparql, query: str):
    #print("query:",query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_all_columns(sparql, table_name: str):
    #print("getting all columns for: {}".format(table_name))
    col_list = []
    res = execute_query(sparql, get_all_columns_query(table_name))
    for r in res["results"]["bindings"]:
        col_list.append(r["column_id"]["value"])

    return col_list


def get_all_profiled_tables(sparql):
    result = []
    res = execute_query(sparql, get_all_profiled_tables_query())
    for r in res["results"]["bindings"]:
        result.append(r["table_name"]["value"])
    return result


def get_similar_relation_tables(sparql, query_table: str, k: int, relation: str, thresh):
    result = []
    res = execute_query(sparql, get_similar_relation_tables_query(query_table, relation))
    for r in res["results"]["bindings"]:
        table1 = r["table_name1"]["value"]
        table2 = r["table_name2"]["value"]
        certainty = r["certainty"]["value"]
        result.append([table1, table2, certainty])
    result = get_top_k_relation_query(result, k, thresh)
    return result

def get_related_columns_between_2_tables(sparql, table1, table2, thresh):
    result = []
    res = execute_query(sparql, get_related_columns_between_2_tables_query(table1, table2, 'semanticSimilarity'))
    for r in res["results"]["bindings"]:
        certainty = r["c"]["value"]
        if float(certainty) > thresh:
            result.append(r["column_id1"]["value"])
        #result.append(r["column_id2"]["value"])
    
    # res = execute_query(sparql, get_joinable_columns_between_2_tables_query(table1, table2, 'inclusionDependency'))
    # for r in res["results"]["bindings"]:
    #     result.append(r["column_id1"]["value"])
    #     #result.append(r["column_id2"]["value"])
        
    
    return result

# def get_joinable_table_paths(sparql, table:str):
#     result = []
#     res = execute_query(sparql, get_joinable_table_paths_query(table))
#     for r in res["results"]["bindings"]:
#         result.append(r["joinable_tables_name"]["value"])
#     return result

def get_related_columns_between_2_tables_j(SPARQL, target_table: str, si:str, thresh):
    result = []
    res = execute_query(SPARQL, get_related_columns_between_2_tables_j_query(target_table, si))
    for r in res["results"]["bindings"]:
        certainty = r["certainty"]["value"]
        if float(certainty) > thresh:
            result.append(r["column_t"]["value"])
    return result

def get_related_columns_between_2_tables_attribute_precision(sparql, table1, table2, thresh):
    result = []
    res = execute_query(sparql, get_related_columns_between_2_tables_query(table1, table2, 'semanticSimilarity'))
    for r in res["results"]["bindings"]:
        certainty = r["c"]["value"]
        if float(certainty) > thresh:
            c1 = r["column_name1"]["value"]
            c2 = r["column_name2"]["value"] 
            result.append([table1, c1, table2, c2])   
    
    return result

def get_related_columns_between_2_tables_j_attribute_precision(SPARQL, target_table: str, si:str, thresh):
    result = []
    res = execute_query(SPARQL, attribute_precision_j_query(target_table, si))
    
    for r in res["results"]["bindings"]:
        certainty = r["certainty"]["value"]
        if float(certainty) > thresh:
            target_t = r["target_table"]["value"]
            target_attr = r["target_attribute"]["value"]
            candidate_t = r["joinable_tables_name"]["value"]
            candidate_attr = r["candidate_attribute"]["value"] 
            result.append([target_t, target_attr, candidate_t, candidate_attr])
    
    return result