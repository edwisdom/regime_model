import math
import itertools

import networkx as nx



def gini_coefficient (model, attribute):
    agent_attribute_list = [getattr(a, attribute) for a in model.grid.get_all_cell_contents()]
    total = sum(agent_attribute_list)
    disparities = 0
    for a_i, a_j in itertools.permutations(agent_attribute_list, 2):
        disparities += math.fabs(a_i - a_j)
    try:
        return disparities/(2*total*model.num_nodes)
    except ZeroDivisionError:
        return 0

def gini_capacity(model):
    return gini_coefficient(model, 'capacity')

def gini_resources(model):
    return gini_coefficient(model, 'resources')

def network_measure(model, measure):
    m = getattr(nx, measure)
    return m(model.G)

def average_clustering(model):
    return network_measure(model, "average_clustering")

def assortativity(model):
    try:
        return network_measure(model, "degree_assortativity_coefficient")
    except ValueError:
        return 0

def number_of_components(model):
    return network_measure(model, "number_connected_components")

def num_satisfied(model):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.satisfied is True])

def num_dissatisfied(model):
    return (model.num_nodes - num_satisfied(model))