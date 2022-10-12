import os

class RDFParser(object):
    def __init__(self, graphs_dir, file_extension):
        graphs_dir = graphs_dir.rstrip('/') + '/'
        self.file_names = [graphs_dir + i for i in os.listdir(graphs_dir) if i.endswith(file_extension)]
        self.all_nodes = set()
        self.all_edges = set()
        self.n_all_triples = 0

    def parse_graphs(self):
        for file_name in self.file_names:
            self._parse_ntriples_file(file_name)

    def print_stats(self):
        print('# Graphs:      ', len(self.file_names))
        print('# Triples:     ', self.n_all_triples)
        print('# Unique Nodes:', len(self.all_nodes))
        print('# Unique Edges:', len(self.all_edges))


class KGLiDSParser(RDFParser):
    """docstring for KGLiDSParser."""

    def __init__(self, graphs_dir, file_extension='txt'):
        super(KGLiDSParser, self).__init__(graphs_dir, file_extension)

    def _parse_ntriples_file(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()

        lines = [i.rstrip() for i in lines]
        self.n_all_triples += len([i for i in lines if i.endswith(';')
                                                    or i.endswith(' .')])

        improper_obj = False
        is_current_multiline_str = False
        current_str_obj = ''
        idx = 2
        while idx < len(lines):
            line = lines[idx]
            idx += 1

            # skip empty lines
            if len(line.strip()) == 0:
                continue

            # rdf star triples have fixed fomrat.
            # read this line and the next to get all nodes and edges.
            elif line.startswith('<<'):
                triple = line[2:line.index('>>')]
                subj, pred, obj = triple.split()[0], triple.split()[1], triple.split()[2]
                star_pred = 'pipeline:parameterValue'
                star_obj = lines[idx][lines[idx].index(star_pred):-2].strip()

                self.all_nodes.add(subj)
                self.all_nodes.add(obj)
                self.all_nodes.add(star_obj)
                self.all_edges.add(pred)
                self.all_edges.add(star_pred)
                idx += 1
                continue

            # a new node and its triples
            elif line.startswith('<'):
                if line.endswith('>'):
                    self.all_nodes.add(line.strip())
                else:
                    # sometimes the predicate is in the same line as node (malformatted)
                    # read this line and the next line
                    pred = line.split()[1]
                    self.all_edges.add(pred)
                    if lines[idx].endswith(';') or lines[idx].endswith(' .'):
                        obj = lines[idx][:-1].strip()
                        self.all_nodes.add(obj)
                    else:
                        is_current_multiline_str = True
                        current_str_obj += line.lstrip()
                    idx += 1
                continue

            # some objects are multi-line strings.
            # if we are currently in a multiline string
            if is_current_multiline_str:
                # if the multi-line string ends in this line
                if line.endswith(';') or line.endswith(' .'):
                    current_str_obj += line[:-1]
                    self.all_nodes.add(current_str_obj.rstrip())
                    # reset
                    is_current_multiline_str = False
                    current_str_obj = ''
                # if the multi-line string continues to next line
                else:
                    current_str_obj += line
                continue

            # we are now reading a line containing a predicate and object
            pred = line.strip().split()[0]
            self.all_edges.add(pred)
            if line.endswith(';') or line.endswith(' .'):
                obj = line[line.index(pred)+len(pred)+1:-1].rstrip()
                self.all_nodes.add(obj)
            # the object continues to next line
            else:
                is_current_multiline_str = True
                current_str_obj += line[line.index(pred)+len(pred)+1:]



class Graph4CodeParser(RDFParser):
    """docstring for Graph4CodeParser."""

    def __init__(self, graphs_dir, file_extension='nq'):
        super(Graph4CodeParser, self).__init__(graphs_dir, file_extension)


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



def main():
    kglids_graphs_dir = '/home/mossad/projects/kglids/pipelines/data/kglids_samples'
    g4c_graphs_dir = '/home/mossad/projects/kglids/pipelines/data/graph4code_samples'
    print('*'*25, 'Parsing KGLiDS', '*'*25)
    parser = KGLiDSParser(kglids_graphs_dir)
    parser.parse_graphs()
    parser.print_stats()
    import re
    print([i for i in parser.all_nodes if i.startswith('<') and (not re.search('\/s[0-9]', i)) and ('/pipeline/library' not in i) and ('http://kglids.org/kaggle/titanic/' not in i) ])

    print('*'*25, 'Parsing Graph4Code', '*'*25)
    parser = Graph4CodeParser(g4c_graphs_dir)
    parser.parse_graphs()
    parser.print_stats()

if __name__ == '__main__':
    main()
