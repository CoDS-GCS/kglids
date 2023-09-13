import io

import pandas as pd
import stardog as sd


def run_query(query_path, db_conn: sd.Connection):
    with open(query_path, 'r') as f:
        query = f.read()

    results = db_conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(results))
    return df


def main():
    connection_details = {
        'endpoint': 'http://localhost:5820',
        'username': 'admin',
        'password': 'admin'
    }
    database_name = 'all_kaggle'
    conn = sd.Connection(database_name, **connection_details)

    #### analysis

    # number of notebooks per year
    num_notebooks_per_year_df = run_query('queries/num_notebooks_per_year.sparql', conn)
    print('Number of notebooks per year:')
    print(num_notebooks_per_year_df)

    # Library use (percentage of notebooks) per year
    raw_lib_use_per_year_df = run_query('queries/library_use_per_year.sparql', conn)
    lib_use_per_year_df = raw_lib_use_per_year_df.merge(num_notebooks_per_year_df, on='year')
    lib_use_per_year_df['perc_year_lib_notebooks'] = lib_use_per_year_df['num_year_lib_notebooks'] * 100 / lib_use_per_year_df['num_year_notebooks']
    print(lib_use_per_year_df)

    lib_use_per_year_df = lib_use_per_year_df[['year', 'library', 'perc_year_lib_notebooks']]
    lib_use_per_year_df.to_csv('library_use_per_year.csv', index=False)


if __name__ == '__main__':
    main()