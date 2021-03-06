"""
	This module initializes graph topologies for the RegimeModel, using
	the create_graph function.
"""

import networkx as nx

# One function per possible network topology
def random_graph(n, p): 
    return nx.erdos_renyi_graph(n=n, p=p)

def small_world(n, p):
    return nx.newman_watts_strogatz_graph(n=n, k=2, p=p)

def scale_free(n, p):
    return nx.barabasi_albert_graph(n=n, m=int(0.5*p*n + 1))


# Initialize shapes to map strings indicating shape to a graph-generator
shapes = dict.fromkeys(['random', 'erdos_renyi', 'erdos', 'er'], random_graph)
shapes.update(dict.fromkeys(['small_world', 'watts_strogatz', 'watts', 'ws'], 
                            small_world))
shapes.update(dict.fromkeys(['scale_free', 'barabasi_albert', 'barabasi', 'ba'], 
                            scale_free))


def create_graph(shape, num_nodes, p):
	""" Returns a Networkx graph.

	Arguments:
	shape -- String indicating the network topology
	num_nodes -- Integer indicating number of nodes
	p -- Float controlling network connectivity, 0 < p <= 1
	"""
	return shapes[shape](num_nodes, p)
