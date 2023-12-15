import pandas as pd

def create_triplets(df, name):

    column_names = df.columns.tolist()
    table_name = name

    column_triplets = [(col_name, "http://kglids.org/ontology/isPartOf", table_name) for col_name in column_names]
    column_triplets_type = [('Column', "http://kglids.org/ontology/isPartOf", 'Table') for col_name in column_names]

    triplets = column_triplets 
    triplets_df = pd.DataFrame(triplets, columns=['Subject', 'Predicate', 'Object'])
    triplets_df.to_csv('OnDemandDataPrep/storage/output_file_'+name+'.csv', index=False)

    triplets_df_type = pd.DataFrame(column_triplets_type, columns=['stype', 'ptype', 'otype'])
    triplets_df_type.to_csv('OnDemandDataPrep/storage/output_type_'+name+'.csv', index=False)
