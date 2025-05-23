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
    num_notebooks_per_year_df.to_csv('num_notebooks_per_year.csv', index=False)

    # Library use (percentage of notebooks) per year
    raw_lib_use_per_year_df = run_query('queries/library_use_per_year.sparql', conn)
    lib_use_per_year_df = raw_lib_use_per_year_df.merge(num_notebooks_per_year_df, on='year')
    lib_use_per_year_df['perc_year_lib_notebooks'] = lib_use_per_year_df['num_year_lib_notebooks'] / lib_use_per_year_df['num_year_notebooks']
    print(lib_use_per_year_df)

    lib_use_per_year_df = lib_use_per_year_df[['year', 'library', 'perc_year_lib_notebooks']]
    # remove URI from library names
    lib_use_per_year_df['library'] = lib_use_per_year_df['library'].apply(
        lambda x: x.replace('http://kglids.org/resource/library/', ''))
    # filter out builtin and util libraries
    filter_out_libs = {'__future__', 'abc', 'ast', 'astropy', 'base64', 'bs4', 'collections', 'csv', 'datetime',
                       'dateutil', 'glob', 'itertools', 'math', 'os', 'random', 're', 'regex', 'requests',
                       'subprocess', 'sys', 'time', 'timeit', 'warnings', 'zipfile', 'operator', 'gc',
                       'pickle', 'multiprocessing', 'copy', 'IPython', 'string', 'io', 'json', 'chardet', 'shutil',
                       'functools', 'pathlib', 'joblib', 'logging', 'tqdm', 'typing'}
    lib_use_per_year_df = lib_use_per_year_df[~lib_use_per_year_df['library'].isin(filter_out_libs)]

    lib_use_per_year_df.to_csv('library_use_per_year.csv', index=False)


if __name__ == '__main__':
    main()