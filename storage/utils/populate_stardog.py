import os
import stardog as sd
from tqdm import tqdm

def poulate_graph4code():
    connection_details = {
      'endpoint': 'http://localhost:5820',
      'username': 'admin',
      'password': 'admin'
    }
    database_name = 'g4c_1000'


    with sd.Admin(**connection_details) as admin:
        if database_name in [db.name for db in admin.databases()]:
            admin.database(database_name).drop()
        db = admin.new_database(database_name, {'edge.properties': True})

    conn = sd.Connection(database_name, **connection_details)

    conn.begin()

    nq_files = ['output_1000_graphs/'+i for i in os.listdir('output_1000_graphs') if i.endswith('nq')]
    for nq in tqdm(nq_files):
        with open(nq, 'r') as f:
            lines = f.readlines()

        if not lines:
            print('Skipping Empty Graph:', nq)
            continue
        first_line = lines[0].strip()
        graph_uri = first_line[first_line.rindex('<'):-2].strip()
        conn.add(sd.content.File(nq), graph_uri=graph_uri)
    conn.commit()



def populate_kglids():
    connection_details = {
      'endpoint': 'http://localhost:5820',
      'username': 'admin',
      'password': 'admin'
    }
    database_name = 'kglids_1000'


    with sd.Admin(**connection_details) as admin:
        if database_name in [db.name for db in admin.databases()]:
            admin.database(database_name).drop()
        db = admin.new_database(database_name, {'edge.properties': True})

    conn = sd.Connection(database_name, **connection_details)

    conn.begin()
    graphs_dir = 'data/graphs/kglids_1000_graphs/'
    ttl_files = [i for i in os.listdir(graphs_dir) if i.endswith('ttl')]
    for ttl in tqdm(ttl_files):
        conn.add(sd.content.File(graphs_dir + ttl), graph_uri= 'http://kglids.org/p/' + ttl[:ttl.rindex('.')])
    conn.commit()


if __name__ == '__main__':
    populate_kglids()
