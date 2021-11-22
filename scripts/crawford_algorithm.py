#!usr/bin/env python 
"""
Crawford Metropolis-within-Gibbs algorithm
"""
from networkx.algorithms.bipartite.basic import degrees
import numpy as np
import networkx as nx
from tqdm import tqdm

class Crawford:
    """
    Class to make inferences based on Crawford paper about a Graphical model
    for Respondent-driven sampling.
    """
    def __init__(self, recruitment_graph, coupon_matrix, times, degrees, n_seeds) -> None:
        self.degrees = degrees
        self.G_R = recruitment_graph
        self.times = times
        self.coupon_matrix = coupon_matrix
        self.n_seeds = n_seeds
        self.n_samples = len(times)
        waiting_times = [times[i+1] - times[i] for i in range(self.n_samples-1)]
        waiting_times.insert(0, 0)
        self.waiting_times = np.array(waiting_times)
        

    def _add_remove_edges(self, G_S):
        """
        Describe a method that from a compatible graph, obtains another.
        """
        A = nx.adjacency_matrix(G_S).toarray()
        u = self.degrees - A.sum(axis=1)
        while True:
            i, j = np.random.choice(list(G_S.nodes), replace=False, size=2)
            has_edge = G_S.has_edge(i, j)
            if not has_edge and u[i] >= 1 and u[j] >= 1:
                G_S.add_edge(i, j)
                return G_S
            elif has_edge and self.G_R.has_edge(i, j):
                G_S.remove_edge(i, j)
                return G_S
            else:
                continue

    def _probability_metropolis_graph(self, rate):
        return min(1,0)

    def _probability_metropolis_rate(self, G_S):
        return min(1,0)

    def graph_given_rate(self, G_S, rate):
        """
        Samples from G_S | lambda.
        """
        G_S_new = self._add_remove_edges(G_S)
        rho = self._probability_metropolis_graph(rate)
        if np.random.random() < rho:
            return G_S_new
        return G_S
    
    def rate_given_graph(self, rate, G_S):
        """
        Approximating the distribution of lambda | G_S.times
        """
        A = nx.adjacency_matrix(G_S)
        u = self.degrees - A.sum(axis=1)
        s = np.sum(np.tril(A@self.coupon_matrix), axis=0) + (self.coupon_matrix.T)@u
        d = np.sum(s * self.waiting_times)
        lambda_hat = (self.n_samples - self.n_seeds) / d
        sigma_hat = lambda_hat / np.sqrt(self.n_samples - self.n_seeds)
        rate_new = np.random.normal(lambda_hat, sigma_hat) 
        rho = self._probability_metropolis_graph(G_S)
        if np.random.random() < rho:
            return rate_new
        return rate              

    def metropolis_within_gibbs(self):
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
        susceptible_edges = {s: {neighbor: (graph.edges[(s, neighbor)])['edge_time']
                                    for neighbor in graph.neighbors(s)}
                                for s in seeds}
        coupon_matrix[np.triu_indices(n_seeds, k=1)] = 1.0
        for new_node in tqdm(range(n_seeds, n_samples)):
            new_time = np.inf
            for r, (rec, info) in enumerate(recruited.items()):
                if info[1] > 0:
                    for s, time in susceptible_edges[rec].items():
                        if s in recruited:
                            continue
                        ev_time = info[0] + time
                        if ev_time < new_time:
                            new_time = ev_time
                            new_participant = s
                            recruiter = rec
                    coupon_matrix[r, new_node] = 1.0
            recruited[new_participant] = [new_time, coupons[new_node]]
            recruited[recruiter][1] -= 1
            susceptible_edges[new_participant] = {neighbor: graph.edges[(new_participant, neighbor)]['edge_time']
                                                            for neighbor in set(graph.neighbors(new_participant))
                                                                            - set(recruited.keys())}
            rds_sample.add_node(new_participant, time=new_time, degree=graph.degree(new_participant))
            rds_sample.add_edge(recruiter, new_participant)

        data = {'rds': rds_sample, 'C': coupon_matrix}
        return data

if __name__ == '__main__':
    simulation = Crawford()
    graph = nx.gnp_random_graph(n=10000, p=0.05, seed=371209)
    data, graph = simulation.simulating_rds_crawford(graph,
                                                     n_samples=500,
                                                     n_seeds=10,
                                                     rate=1,
                                                     probs=[1/3, 1/6, 1/6, 1/3],
                                                     seed=371209)
