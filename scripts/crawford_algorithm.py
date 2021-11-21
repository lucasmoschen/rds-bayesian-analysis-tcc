#!usr/bin/env python 
"""
Crawford Metropolis-within-Gibbs algorithm
"""
import numpy as np
import networkx as nx

class Crawford:
    """
    Class to make inferences based on Crawford paper about a Graphical model
    for Respondent-driven sampling.
    """
    def __init__(self) -> None:
        pass

    def simulating_rds_crawford(self, graph, n_samples, n_seeds, rate, probs,
                                seed):
        """
        Simulate RDS from Crawford model.
        """
        ro = np.random.RandomState(seed)
        exponential_times = ro.exponential(scale=rate, size=graph.number_of_edges())
        exponential_times = {edge: exponential_times[i] for i, edge in enumerate(graph.edges)}
        nx.set_edge_attributes(graph, exponential_times, name='edge_time')

        degrees = dict(graph.degree)
        nx.set_node_attributes(graph, degrees, name='degree')
        coupons = ro.choice(range(4), p=probs, size=n_samples)

        degree_prob = np.array(list(degrees.values()))
        degree_prob = 1.0 * degree_prob / degree_prob.sum()
        seeds = ro.choice(list(graph.nodes), p=degree_prob, size=n_seeds)



if __name__ == '__main__':
    simulation = Crawford()
    graph = nx.gnp_random_graph(n=1000, p=0.05)
    simulation.simulating_rds_crawford(graph, 10, 10, 1, [1/3, 1/6, 1/6, 1/3], 371209)
