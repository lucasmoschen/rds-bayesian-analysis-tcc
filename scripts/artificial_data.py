#!/usr/bin/env python

import networkx as nx
import numpy as np

class GenerateData: 

    def __init__(self,): 
        pass

    def random_graph(self, rule, parameters): 
        """
        Generates a random graph according to the specified rule. 
        - rule (string): name of the random graph generator; 
        - parameters (dict): necessary parameters to the model. 
        """
        if rule == 'erdos-renyi':   
            n = parameters['n']
            p = parameters['p']
            seed = parameters['seed']
            G = nx.gnp_random_graph(n, p, seed)
        else: 
            raise Exception("This rule was not developed yet. ")

        return G

    def RDS_generator(self, graph, seed, n_seeds, sample_size, probs):
        """
        This function is a way of generating artificial subgraph of
        recruitment when a population graph is specified.
        """
        ro = np.random.RandomState(seed = seed)
        seeds = ro.choice(graph.nodes, size = n_seeds, replace = False)

        recruited = set(seeds)
        current_recruited = set()
        edges = []

        while len(recruited) < sample_size: 
            for node in seeds:
                print(node)
                neighbors = set(graph.neighbors(node)) - recruited
                s = ro.choice([0,1,2,3], p = probs)
                if len(neighbors) == 0 or s == 0:
                    continue 
                elif len(neighbors) < s: 
                    s = len(neighbors)
                recruits = ro.choice(list(neighbors), size = s, replace = False)
                for r in recruits: 
                    edges.append((node, r))
                
                recruited = recruited.union(recruits)
                current_recruited = current_recruited.union(recruits)

            seeds = current_recruited
            current_recruited = set()

        G = nx.DiGraph() 
        G.add_nodes_from(recruited)
        G.add_edges_from(edges)

        return G

if __name__ == '__main__': 

    gen_graph = GenerateData()
    graph = gen_graph.random_graph(rule = 'erdos-renyi', parameters={'n': 1000, 'p': 0.5, 'seed': 10000})
    rds_sample = gen_graph.RDS_generator(graph = graph, seed = 20000, 
                                     n_seeds = 4, sample_size = 10, probs = [1/3,1/6,1/6,1/3])
                                     
    