import os

class Graph4CodeParser(object):
    """docstring for Graph4CodeParser."""

    def __init__(self, graphs_dir):
        graphs_dir = graphs_dir.rstrip('/') + '/'
        self.file_names = [graphs_dir + i for i in os.listdir(graphs_dir) if i.endswith('.nq')]
        self.all_nodes = set()
        self.all_edges = set()
        self.n_all_triples = 0

    def parse_graphs(self):
        for file_name in self.file_names:
            self._parse_ntriples_file(file_name)

    def _parse_ntriples_file(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()

        lines = [i.strip() for i in lines]
        self.n_all_triples += len(lines)

        graph_name = lines[0][lines[0].index('<http://github/'):].split()[0]

        for idx, line in enumerate(lines):
            assert line.startswith('<'), f"Error parsing file: {file_name}:\nLine {i+1} should start with <"
            triple = line[:line.index(graph_name)].strip()
            if triple.startswith('<<'):
                # rdf-star
                star_triple = triple[2:triple.index('>>')]
                subj, pred = star_triple.split()[0], star_triple.split()[1]
                obj = star_triple[star_triple.index(pred)+len(pred):].strip()
                star_pred = triple[triple.index('>>')+2:].split()[0]
                star_obj = triple[triple.index(star_pred)+len(star_pred):].strip()

                self.all_nodes.add(subj)
                self.all_nodes.add(obj)
                self.all_nodes.add(star_obj)
                self.all_edges.add(pred)
                self.all_edges.add(star_pred)

            else:
                subj, pred = triple.split()[0], triple.split()[1]
                obj = triple[triple.index(pred) + len(pred):].strip()

                self.all_nodes.add(subj)
                self.all_nodes.add(obj)
                self.all_edges.add(pred)

    def print_stats(self):
        print('# Graphs:      ', len(self.file_names))
        print('# Triples:     ', self.n_all_triples)
        print('# Unique Nodes:', len(self.all_nodes))
        print('# Unique Edges:', len(self.all_edges))


def main():
    graphs_dir = '/home/mossad/projects/kglids/pipelines/data/graph4code_samples'
    parser = Graph4CodeParser(graphs_dir)
    parser.parse_graphs()
    parser.print_stats()


if __name__ == '__main__':
    main()
