#!usr/bin/env python 
"""
Crawford Metropolis-within-Gibbs algorithm
"""
import numpy as np
import networkx as nx
from tqdm import tqdm

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
        coupons = ro.choice(range(4), p=probs, size=n_samples)
        coupon_matrix = np.zeros((n_samples, n_samples))

        degree_prob = np.array(list(degrees.values()))
        degree_prob = 1.0 * degree_prob / degree_prob.sum()
        seeds = ro.choice(list(graph.nodes), p=degree_prob, size=n_seeds,
                          replace=False)

        rds_sample = nx.DiGraph()
        recruitment_seed = ro.exponential(scale=rate, size=n_seeds)
        recruitment_seed.sort()
        recruited = {seeds[i]: [recruitment_seed[i], coupons[i]] for i in range(n_seeds) if coupons[i] > 0}
        n_seeds = len(recruited)
        for node in recruited:
            rds_sample.add_node(node, time=recruited[node][0], degree=graph.degree(node))
        susceptible_edges = {s: {neighbor: graph.edges[(s, neighbor)]['edge_time']
                                    for neighbor in graph.neighbors(s)}
                                for s in seeds}
        coupon_matrix[np.triu_indices(n_seeds, k=1)] = 1.0
        for new_node in tqdm(range(n_seeds, n_samples)):
            new_time = np.inf
            for r, (rec, info) in enumerate(recruited.items()):
                if info[1] > 0:
                    for s, time in susceptible_edges[rec].items():
                        ev_time = info[0] + time
                        if ev_time < new_time:
                            new_time = ev_time
                            new_participant = s
                            recruiter = rec
                    coupon_matrix[r, new_node] = 1.0
            recruited[new_participant] = [new_time, coupons[new_node]]
            recruited[recruiter][1] -= 1
            popped = susceptible_edges[recruiter].pop(new_participant, 0)
            if popped == 0:
                print('INFO - The recruitments stopped because no one wanted to recruit anymore.')
                coupon_matrix = coupon_matrix[:new_node, :new_node]
                break
            susceptible_edges[new_participant] = {neighbor: graph.edges[(s, neighbor)]['edge_time']
                                                            for neighbor in graph.neighbors(s)}
            rds_sample.add_node(new_participant, time=new_time, degree=graph.degree(new_participant))
            rds_sample.add_edge(recruiter, new_participant)

        data = {'rds': rds_sample, 'C': coupon_matrix}
        return data, graph

if __name__ == '__main__':
    simulation = Crawford()
    graph = nx.gnp_random_graph(n=10000, p=0.05, seed=371209)
    data, graph = simulation.simulating_rds_crawford(graph,
                                                     n_samples=500,
                                                     n_seeds=10,
                                                     rate=1,
                                                     probs=[1/3, 1/6, 1/6, 1/3],
                                                     seed=371209)
