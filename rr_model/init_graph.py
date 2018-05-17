import networkx as nx


def random_graph(n, p): 
    return nx.erdos_renyi_graph(n=n, p=p)

def small_world(n, p):
        return nx.newman_watts_strogatz_graph(n=n, k=2, p=p)

def scale_free(n, p):
        return nx.barabasi_albert_graph(n=n, m=0.5*p*n + 1)


shapes = dict.fromkeys(['random', 'erdos_renyi', 'erdos', 'er'], random_graph)
shapes.update(dict.fromkeys(['small_world', 'watts_strogatz', 'watts', 'ws'], 
                            small_world))
shapes.update(dict.fromkeys(['scale_free', 'barabasi_albert', 'barabasi', 'ba'], 
                            scale_free))


def create_graph(shape, num_nodes, p):    
    return shapes[shape](num_nodes, p)
