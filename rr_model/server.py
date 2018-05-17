import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from .model import RegimeModel, num_satisfied, num_dissatisfied, gini_resources, gini_capacity


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_size(agent):
        return agent.capacity

    def node_color(agent):
        return {
            False: '#FF0000',
            True: '#008000'
        }.get(agent.satisfied, '#808080')

    def edge_width(source, target):
        return 10 * G.edges[(source,target)]['capacity_t']

    def get_agents(source, target):
        return G.node[source]['agent'][0], G.node[target]['agent'][0]

    portrayal = dict()
    portrayal['nodes'] = [{'size': node_size(agents[0]),
                           'color': node_color(agents[0]),
                           'tooltip': "ID: {}<br>Resources: {}".format(str(agents[0].unique_id), str(agents[0].resources)),
                           }
                          for (_, agents) in G.nodes.data('agent')]

    portrayal['edges'] = [{'source': source,
                           'target': target,
                           'color': '#000000',
                           'width': edge_width(source, target),
                           }
                          for (source, target) in G.edges]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500, library='d3')
chart = ChartModule([{'Label': 'Dissatisfied', 'Color': '#FF0000'},
                     {'Label': 'Satisfied', 'Color': '#008000'}])


class MyTextElement(TextElement):
    def render(self, model):
        gini_c_text = str(gini_capacity(model))
        gini_r_text = str(gini_resources(model))
        
        return "Gini Coefficient of Capacity: {}<br>Gini Coefficient of Resources: {}".format(gini_c_text, gini_r_text)


model_params = {
    'num_nodes': UserSettableParameter('slider', 'Number of Agents', 10, 10, 100, 1,
                                       description='Choose how many agents to include in the model'),
    'productivity': UserSettableParameter('slider', 'Productivity (Shock Variable)', 0.5, 0.05, 1, 0.05,
                                          description='Set environmental productivity'),
    'demand': UserSettableParameter('slider', 'Demand', 3, 1, 10, 0.1,
                                    description='Agent per-turn demand'),
    'network_parameter': UserSettableParameter('slider', 'Network Parameter', 0.3, 0.0, 1.0, 0.05,
                                                 description='Probability of a link'),
    'resource_inequality': UserSettableParameter('slider', 'Resource Inequality', 0.5, 0.1, 2.0, 0.1,
                                                   description='Initial resource inequality'),
    'capacity_inequality': UserSettableParameter('slider', 'Capacity Inequality', 0.5, 0.1, 2.0, 0.1,
                                             description='Initial capacity inequality'),
    'uncertainty': UserSettableParameter('slider', 'Transaction Uncertainty', 0.2, 0.1, 2.0, 0.1,
                                                    description='Uncertainty about transaction costs'),
}

server = ModularServer(RegimeModel, [network, MyTextElement(), chart], 'Regime Model', model_params)
server.port = 8521
