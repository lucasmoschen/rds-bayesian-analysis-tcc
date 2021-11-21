#!/usr/bin/env python

import networkx as nx
import numpy as np
from tqdm.notebook import tqdm

class GenerateData: 

    def __init__(self,): 
        pass

    def random_graph(self, rule, parameters): 
        """
        Generates a random graph according to the specified rule. 
        - rule (string): name of the random graph generator; 
        - parameters (dict): necessary parameters to the model. 
        """
        n = parameters['n']
        seed = parameters['seed']
        if rule == 'erdos-renyi':
            p = parameters['p']
            G = nx.gnp_random_graph(n, p, seed)
        elif rule == 'erdos-renyi-preferential':
            p1 = parameters['p1']
            p2 = parameters['p2']
            p_z1 = parameters['p_z1']
            ro = np.random.RandomState(seed)
            Z = ro.binomial(n=1, p=p_z1, size=n)
            nodes = [(k, {'binary': Z[k]}) for k in range(n)]
            edges = []
            for i in tqdm(range(n)):
                for j in range(i+1, n):
                    if Z[i] == Z[j]:
                        if ro.random() < p1:
                            edges.append((i, j))
                    else:
                        if ro.random() < p2:
                            edges.append((i, j))
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        elif rule == 'barabasi-albert':
            m = parameters['m']
            G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
        elif rule == 'barabasi-albert-preferential':
            p = parameters['p']
            m = parameters['m']
            p_z1 = parameters['p_z1']
            ro = np.random.RandomState(seed)
            Z = ro.binomial(n=1, p=p_z1, size=n)
            G = nx.gnp_random_graph(m, 0.9, seed)
            for node in range(m):
                nx.set_node_attributes(G, Z[node], 'binary')
            for node in tqdm(range(m, n)):
                degrees = np.array(list(G.degree))[:, 1]
                probs = np.zeros(node)
                x = Z[:node]
                probs[x == Z[node]] = p * degrees[x == Z[node]]/degrees[x == Z[node]].sum()
                probs += (1 - p) * degrees/degrees.sum()
                chosen_nodes = ro.choice(range(node), replace=False, size=m)
                G.add_nodes_from([(node, {'binary': Z[node]})])
                G.add_edges_from([(node, k) for k in chosen_nodes])
        else:
            raise Exception("This rule was not developed yet. ")

        return G

    def RDS_generator(self, graph, seed, n_seeds, sample_size, probs, R=0):
        """
        This function is a way of generating artificial subgraph of
        recruitment when a population graph is specified.
        """
        ro = np.random.RandomState(seed=seed)
        degrees = np.array(list(dict(graph.degree()).values()))
        seeds = ro.choice(graph.nodes, size=n_seeds, replace=False,
                          p=degrees/degrees.sum())

        recruited = set(seeds)
        current_recruited = set()
        edges = []

        while len(recruited) < sample_size:
            for node in seeds:
                neighbors = set(graph.neighbors(node)) - recruited
                s = ro.choice([0, 1, 2, 3], p=probs)
                if len(neighbors) == 0 or s == 0:
                    continue
                elif len(neighbors) < s:
                    s = len(neighbors)
                probs_nei = np.zeros(len(neighbors))
                R_x = R
                if R > 0:
                    Z = np.array([x[1]['binary'] for x in list(graph.nodes.data())])
                    neighbors_array = np.array(list(neighbors))
                    a = Z[neighbors_array]
                    sum_a = (a == 1).sum()
                    if sum_a > 0:
                        probs_nei[a == 1] = R/sum_a
                        R_x = R
                    else: R_x = 0
                probs_nei += (1-R_x)/len(neighbors)
                recruits = ro.choice(list(neighbors), size=s, replace=False,
                                     p=probs_nei)
                for r in recruits:
                    edges.append((node, r))
                
                recruited = recruited.union(recruits)
                current_recruited = current_recruited.union(recruits)

            seeds = current_recruited
            current_recruited = set()

        G = nx.DiGraph()
        G.add_nodes_from(recruited)
        G.add_edges_from(edges)

        if R > 0:
            binary = {node: graph.nodes[node]['binary'] for node in G.nodes}
            nx.set_node_attributes(G, binary, 'binary')
        return G

if __name__ == '__main__':

    gen_graph = GenerateData()
    graph = gen_graph.random_graph(rule='erdos-renyi', parameters={'n': 1000, 'p': 0.5, 'seed': 10000})
    rds_sample = gen_graph.RDS_generator(graph = graph, seed = 20000, 
                                     n_seeds = 4, sample_size = 10, probs = [1/3,1/6,1/6,1/3])
                                     
    