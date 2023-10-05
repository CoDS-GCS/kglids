import pandas as pd

def get_table_mapping(df: pd.DataFrame):
    print("getting mapping between tables from ground truth")
    query_tables = df[df.columns[0]].to_list()
    candidate_tables = df[df.columns[1]].to_list()
    mapping = {(query_tables[i], candidate_tables[i]) for i in range(len(query_tables))}

    return mapping