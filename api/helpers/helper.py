import stardog


def connect_to_stardog(db_name: str = 'kglids'):
    connection_details = {
        'endpoint': 'http://localhost:5820',
        'username': 'admin',
        'password': 'admin'
    }
    print('Connected to https://cloud.stardog.com/u/0/studio/#/')
    return stardog.Connection('kglids', **connection_details)


def execute_query(conn: stardog.Connection, query):
    return conn.select(query, content_type='text/csv')
