from random import random, expovariate, paretovariate
import math
from enum import Enum
import itertools
from operator import attrgetter

import networkx as nx

# Use agent-based modeling library, Mesa
from mesa import Agent, Model
from mesa.time import StagedActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# Use modules to initialize network and report data
from init_graph import create_graph
from reporters import gini_capacity, gini_resources
from reporters import average_clustering, assortativity, number_of_components
from reporters import num_satisfied, num_dissatisfied


def calculate_transfer(p, c_1, c_2, t):
    """Calculates the resources to be transferred over a link."""

    c_hi, c_lo = (c_1, c_2) if c_1 > c_2 else (c_2, c_1)
    try:
        patron_gain = math.exp(p*(c_hi+t)) - math.exp(p*c_hi)
        client_loss = math.exp(p*c_lo) - math.exp(p*(c_lo-t))
        return math.sqrt(patron_gain*client_loss) # Geometric mean
    except OverflowError:
        client_loss = math.exp(p*c_lo) - math.exp(p*(c_lo-t))
        return math.sqrt(c_hi)*client_loss


def utility(d, r, c):
    """
        Calculates the utility function for a given level of demand,
        resources, and capacity
    """
    try:    
        resource_u = 2/(1+math.exp(-r + d))
    except OverflowError:
        resource_u = 2
    try:
        capacity_u = (math.sinh(c)/math.sinh(d))
    except OverflowError:
        capacity_u = 1e6
    return resource_u*capacity_u


def delta_u(d, r_old, c_old, r_new, c_new):
    """Returns the utility differential of new r,c-values over old ones"""
    return (utility(d, r_new, c_new) - utility(d, r_old, c_old))


def utility_gain(d, r, c, resource_t, capacity_t, is_patron, is_adding):
    """Returns the utility gain in performing some action

    Arguments:
    d -- Demand
    r -- Resources
    c -- Capacity
    resource_t -- The expected resource transfer
    capacity_t -- The expected capacity transfer
    is_patron -- A bool indicating whether the actor is a patron
    is_adding -- A bool indicating whether we're adding a link 
    """

    # Patrons, when adding links, gain capacity and lose resources. (0)
    # Clients, when adding links, lose capacity and gain resources. (1)
    # Patrons, when removing links, lose capacity and gain resources. (1)
    # Clients, when removing links, gain capacity and lose resources. (0)
    p_i = 2*(is_patron^is_adding) - 1 # P_i is 1 in case 1, and -1 in case 0
    return (delta_u(d, r, c, r+(p_i*resource_t), c-(p_i*capacity_t)))


def average(a_list):
    """Returns an arithmetic average of a list of numbers"""
    return sum(a_list)/len(a_list)


class RegimeModel(Model):
    # A model of a regime

    def __init__(self, num_nodes, productivity, demand, shape, network_param,
                 resource_inequality, capacity_inequality, uncertainty, shock):

        # Initialize graph
        self.num_nodes = num_nodes
        self.G = create_graph(shape, num_nodes, network_param)
        self.G = max(nx.connected_component_subgraphs(self.G), 
                     key=lambda g: len(g.nodes()))
        
        # Initialize other attributes
        self.grid = NetworkGrid(self.G)
        self.schedule = StagedActivation(self, 
                                         stage_list=['add_link', 'cut_link',
                                                    'settle_env_transfer', 
                                                    'settle_link_transfer'
                                                    ],
                                         shuffle=True,
                                         shuffle_between_stages=True)
        self.productivity = productivity
        self.demand = demand
        self.resource_inequality = resource_inequality
        self.capacity_inequality = capacity_inequality
        self.uncertainty = uncertainty
        self.datacollector = DataCollector({
                                "Gini Coeff. of Capacity": gini_capacity,
                                "Gini Coeff. of Resources": gini_resources,
                                "Satisfied": num_satisfied,
                                "Dissatisfied": num_dissatisfied
                            })
        self.shock = shock

        # Initialize agents on graph (cap capacity at 25 to avoid overflow errors)
        for i, node in enumerate(self.G.nodes()):
            a = RegimeAgent(i, self, self.demand, 
                            math.exp(paretovariate(1/resource_inequality)),
                            min(25, paretovariate(1/capacity_inequality)),
                            True)
            self.schedule.add(a)
            self.grid.place_agent(a, node)

            
        for e in self.G.edges():
            capacity_transfer = expovariate(1/self.uncertainty)
            agents = ([self.G.nodes[e[0]]['agent'][0], 
                     self.G.nodes[e[1]]['agent'][0]])
            resource_transfer = calculate_transfer(self.productivity,
                                agents[0].capacity + capacity_transfer,
                                agents[1].capacity + capacity_transfer,
                                capacity_transfer)
            
            # Give the agent with the higher capacity 2*transfer capacity to
            # simulate a prior interaction where the transfer was made, when
            # both the nodes had their current capacity + transfer.
            max_capacity_agent = max(agents, key=attrgetter('capacity'))
            min_capacity_agent = min(agents, key=attrgetter('capacity'))
            max_capacity_agent.capacity += 2*capacity_transfer
            min_capacity_agent.exp_resources += resource_transfer

            # Initialize edge attributes for transfer rates, memory of 
            # resource transfers, and which agent is the patron.
            self.G.edges[e]['capacity_t'] = capacity_transfer
            self.G.edges[e]['resource_t'] = resource_transfer
            self.G.edges[e]['exp_resources'] = [resource_transfer, resource_transfer]
            self.G.edges[e]['patron'] = max_capacity_agent
            
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            if i == n//2: 
                # Halfway through simulation, add shock.
                self.productivity += self.shock
            self.step()


class RegimeAgent(Agent):
    def __init__(self, unique_id, model, demand, resources, 
                capacity, satisfied):
        super().__init__(unique_id, model)

        self.demand = demand
        self.resources = resources
        self.capacity = capacity
        self.exp_resources = 0
        self.satisfied = True

    def neighbor_of_neighbors(self):
        """ Returns neighbors of neighbors of an agent """
        neighbors = self.model.grid.get_neighbors(self.pos)
        two_neighbors = set()
        
        # Add all neighbors-of-neighbors to our two-neighbors set
        for n in neighbors:
            neighbors_of_neighbors = self.model.grid.get_neighbors(n)
            two_neighbors = two_neighbors.union(neighbors_of_neighbors) 
        
        # Return only those neighbors-of-neighbors that aren't neighbors
        return list(two_neighbors.difference(neighbors))

    def expected_transfer(self):
        """Returns the expected transfer cost by an agent"""
        links = self.model.G.edges(self.pos)
        transfers = 0
        for l in links:
            transfers += self.model.G.edges[l]['capacity_t']
        # Average all link transfers to create an estimate
        try:
            return transfers/len(links)
        except ZeroDivisionError:
            return expovariate(1/self.model.uncertainty)

    def link_resources(self):
        """Calculates how many resources an agent expects from links"""
        links = self.model.G.edges(self.pos)
        exp_resource_gain = 0
        for l in links:
            # Take an average of the last two transactions for expectation
            resource_t = average(self.model.G.edges[l]['exp_resources'])
            r_gain = (-resource_t if self.model.G.edges[l]['patron'] == self
                                 else resource_t)
            exp_resource_gain += r_gain
        return exp_resource_gain

    def ranked_candidates(self, candidates, capacity_t):
        """
        Returns a ranked list of candidates where it would be a 
        positive-utility action to form a link.

        Arguments:
        candidates -- List of possible candidates (neighbors-of-neighbors)
        capacity_t -- The agent's expected capacity transfer
        """
        delta_utilities = []
        for c in candidates:
            resource_t = calculate_transfer(self.model.productivity, self.capacity,
                                            c.capacity, capacity_t)
            is_patron = (self.capacity > c.capacity)
            is_adding = True
            delta_u = utility_gain(self.demand, self.exp_resources, 
                                   self.capacity, resource_t, capacity_t,
                                   is_patron, is_adding)
            delta_utilities.append(delta_u)
        # Filter out positive utilities and then sort agents by utility
        pos_utilities = [(a,u) for (a,u) in zip(candidates, delta_utilities) if u>0]
        ranked_agents = list(reversed([a for (a,u) in sorted(pos_utilities, key=lambda x: x[1])]))
        return ranked_agents

    def initialize_link(self, other, capacity_t):
        """
        If initializing a link with a transfer size of capacity_t
        between self and other agent is possible, the function does so
        and returns True. Else, it returns false.
        """
        patron, client = ((self, other) if self.capacity > other.capacity 
                                       else (other, self))
        resource_t = calculate_transfer(self.model.productivity, patron.capacity,
                                        client.capacity, capacity_t)
        if resource_t < patron.resources and capacity_t < client.capacity:
            # Change agent's properties once link is formed, and initialize
            # edge attributes. 
            link = (patron.pos, client.pos)
            self.model.G.add_edge(*link)
            patron.capacity += capacity_t
            patron.exp_resources -= resource_t
            client.capacity -= capacity_t
            client.exp_resources += resource_t
            self.model.G.edges[link]['capacity_t'] = capacity_t
            self.model.G.edges[link]['resource_t'] = resource_t
            self.model.G.edges[link]['exp_resources'] = [resource_t, resource_t]
            self.model.G.edges[link]['patron'] = patron
            return True
        else:
            return False

    def add_link(self):
        """
        Stage 1/4 of each simulation time-step: Agents get to add 1 link max.

        Rank neighbor-of-neighbors by the utility of adding a link to them,
        if they 'agree' to a link (utility > 0), try to add it.
        """
        self.exp_resources = max(0, (self.resources + self.link_resources() 
            - self.demand + math.exp(self.model.productivity * self.capacity)))
        two_neighbors = self.neighbor_of_neighbors()
        neighbor_agents = self.model.grid.get_cell_list_contents(two_neighbors)
        exp_capacity_t = self.expected_transfer()
        ranked_agents = self.ranked_candidates(neighbor_agents, exp_capacity_t)
        for a in ranked_agents:
            resource_t = calculate_transfer(a.model.productivity, a.capacity,
                                            self.capacity, exp_capacity_t)
            is_patron = a.capacity > self.capacity
            is_adding = True
            delta_u = (utility_gain(a.demand, a.exp_resources, a.capacity, 
                                    resource_t, exp_capacity_t,
                                    is_patron, is_adding))
            capacity_t = expovariate(1/self.model.uncertainty)
            # Once we've added a link, the stage ends
            if delta_u > 0 and self.initialize_link(a, capacity_t):
                break

    def remove_edge(self, edge, self_patron):
        """Remove an edge between self and other and undo the transaction."""
        other_pos = edge[1] if edge[0] == self.pos else edge[0]
        other_agent = self.model.grid.get_cell_list_contents([other_pos])[0]
        p_i = 2*(self_patron) - 1 # 1 if self is patron, -1 otherwise
        capacity_t = p_i*self.model.G.edges[edge]['capacity_t']
        resource_t = p_i*(average(self.model.G.edges[edge]['exp_resources']))
        other_agent.capacity += capacity_t
        # Note that capacity gets transferred back, but resources don't get
        # "repaid." Any resources sent over a link are unrecoverable. 
        other_agent.exp_resources -= resource_t
        self.capacity -= capacity_t
        self.exp_resources += resource_t
        self.model.G.remove_edge(*edge)

    def cut_link(self):
        """
        Stage 2/4 of each simulation time-step. Agents get to cut 1 link max.

        If an agent has more than 1 link, and they find that one lowers their
        utility, they cut it. The agents search their links in random order.
        """
        edges = self.model.G.edges(self.pos)
        if len(edges) > 1:
            for e in edges:
                # Agents use their memory of the transaction history to
                # evaluate the utility of a link, not the value of a
                # promised transaction.
                resource_t = average(self.model.G.edges[e]['exp_resources'])
                capacity_t = self.model.G.edges[e]['capacity_t']
                is_patron = (self.model.G.edges[e]['patron'] == self)
                is_adding = False
                u_removal = utility_gain(self.demand, self.exp_resources, 
                                      self.capacity, resource_t, capacity_t,
                                      is_patron, is_adding)
                if u_removal > 0:
                    self.remove_edge(e, is_patron)
                    break

    def settle_env_transfer(self):
        """Stage 3/4 of each simulation time-step: Environmental transfers"""
        self.resources = (math.exp(self.model.productivity*self.capacity)
                          - self.demand)
        self.satisfied = self.resources > 0
        # Resources cannot become negative, so we reset them to 0, and 
        # simply indicate dissatisfaction for not meeting demand. 
        self.resources = self.satisfied*self.resources

    def pay_client(self, edge):
        """Pay the client on the other side of an edge"""
        client_pos = edge[1] if edge[0] == self.pos else edge[0]
        client = self.model.grid.get_cell_list_contents([client_pos])[0]
        resources_to_pay = self.model.G.edges[edge]['resource_t']
        # If possible, pay what the client is owed. Else, pay what you can.
        resources_paid = (resources_to_pay if self.resources > resources_to_pay  
                                           else self.resources)
        self.resources -= resources_paid
        client.resources += resources_paid
        self.model.G.edges[edge]['exp_resources'] = ([resources_paid, 
                        self.model.G.edges[edge]['exp_resources'][0]])

    def settle_link_transfer(self):
        """Stage 4/4 of each simulation time-step: Pay all clients"""
        for e in self.model.G.edges(self.pos):
            if self.model.G.edges[e]['patron'] == self:
                self.pay_client(e)

