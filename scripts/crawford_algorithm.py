#!usr/bin/env python
"""
Crawford Metropolis-within-Gibbs algorithm
"""
import numpy as np
import networkx as nx
from tqdm import tqdm

def simulating_rds_crawford(graph, n_samples, n_seeds, rate, probs, seed):
    """
    Simulate RDS from Crawford model.
    """
    ro = np.random.RandomState(seed)
    exponential_times = ro.exponential(scale=1/rate, size=graph.number_of_edges())
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
    recruitment_seed = ro.exponential(scale=1/rate, size=n_seeds)
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

class Crawford:
    """
    Class to make inferences based on Crawford paper about a Graphical model
    for Respondent-driven sampling.
    """
    def __init__(self, recruitment_graph, coupon_matrix, times, degrees) -> None:
        self.degrees = np.array(degrees).reshape(-1, 1)
        self.G_R = recruitment_graph.to_undirected()
        self.adj_matrix = nx.adjacency_matrix(self.G_R).toarray()
        self.nodes = self.G_R.nodes()
        self.times = np.array(times).reshape(-1, 1)
        self.coupon_matrix = coupon_matrix
        self.n_samples = len(times)
        waiting_times = [times[i+1] - times[i] for i in range(self.n_samples-1)]
        waiting_times.insert(0, 0)
        self.waiting_times = np.array(waiting_times).reshape(-1, 1)
        self.non_seeds = np.array([node for node, degree in dict(recruitment_graph.in_degree()).items() if degree > 0])
        self.n_seeds = self.n_samples - len(self.non_seeds)
        self.times_star = self.times + self.coupon_matrix @ self.waiting_times

    def _add_remove_edges(self, G_S, u):
        """
        Describe a method that from a compatible graph, obtains another.
        """
        u_new = np.copy(u)
        G_S_new = G_S.copy()
        while True:
            i, j = np.random.choice(self.nodes, replace=False, size=2)
            has_edge = G_S_new.has_edge(i, j)
            if not has_edge and u[i] >= 1 and u[j] >= 1:
                G_S_new.add_edge(i, j)
                u_new[i] -= 1
                u_new[j] -= 1
                return G_S_new, i, j, u, 'add'
            elif has_edge and not self.G_R.has_edge(i, j):
                G_S_new.remove_edge(i, j)
                u_new[i] += 1
                u_new[j] += 1
                return G_S_new, i, j, u_new, 'remove'
            else:
                continue

    def _proposal_ratio(self, G_S, u):
        """
        Proposal ration calculation
        """
        A = nx.adjacency_matrix(G_S).toarray()
        eval_u = np.array(u >= 1)
        add_GS = np.sum((np.triu(1 - A, k=1) * eval_u) @ eval_u)
        rem_GS = np.sum(A * (1 - self.adj_matrix))//2
        return add_GS, rem_GS

    def _probability_metropolis_graph(self, rate, s_new, s, u_new, u, i, j, 
                                      add_GS, rem_GS, add_GS_new, rem_GS_new, action):
        """
        The acceptance probability for G_S | lambda
        """
        likelihood_ratio = np.sum((np.log(s_new[self.non_seeds]) - np.log(s[self.non_seeds])))
        exponent = self.times_star[i] - min(self.times[j], self.times_star[i]) + self.times_star[j] - self.times[j]
        if action == 'add':
            likelihood_ratio += rate * exponent[0]
        elif action == 'remove':
            likelihood_ratio += -rate * exponent[0]
        likelihood_ratio = np.exp(likelihood_ratio)

        p = likelihood_ratio * (add_GS + rem_GS) / (add_GS_new + rem_GS_new)

        return min(1, p)

    def _probability_metropolis_rate(self, rate, rate_new, d, lambda_hat, sigma_hat, alpha, beta):
        """
        Probability of acceptance for the lambda parameter
        """
        likelihood_ratio = (self.n_samples - self.n_seeds) * (np.log(rate_new) - np.log(rate))
        likelihood_ratio += -d * (rate_new - rate)
        sigma_star = rate_new/np.sqrt(self.n_samples - self.n_seeds)
        propose_ratio = np.log(rate_new) - np.log(rate)
        propose_ratio -= 0.5*((rate - lambda_hat)**2/sigma_hat**2 - (rate_new - lambda_hat)**2/sigma_star**2)
        prior_ratio = (alpha - 1) * (np.log(rate_new) - np.log(rate)) - beta * (rate_new - rate)
        log_p = likelihood_ratio + propose_ratio + prior_ratio
        return np.exp(min(0, log_p))

    def sample_graph_given_rate(self, G_S, rate, u, s, add_GS, rem_GS):
        """waiting_times
        Samples from G_S | lambda.
        """
        G_S_new, i, j, u_new, action = self._add_remove_edges(G_S, u)
        add_GS_new, rem_GS_new = self._proposal_ratio(G_S_new, u_new)
        s_new = self._precalculations(s, i, j, action)
        rho = self._probability_metropolis_graph(rate, s_new, s, u_new, u, i, j,
                                                 add_GS, rem_GS, add_GS_new, rem_GS_new, action)
        if np.random.random() < rho:
            return G_S_new, s_new, u_new, add_GS_new, rem_GS_new
        return G_S, s, u, add_GS, rem_GS
    
    def sample_rate_given_graph(self, rate, s, alpha, beta):
        """if action == 'add':
        Approximating the distribution of lambda | G_S.times
        """
        d = np.dot(s.flatten(), self.waiting_times.flatten())
        lambda_hat = (self.n_samples - self.n_seeds) / d
        sigma_hat = rate / np.sqrt(self.n_samples - self.n_seeds)
        rate_new = np.random.normal(lambda_hat, sigma_hat)
        rho = self._probability_metropolis_rate(rate, rate_new, d, lambda_hat, sigma_hat, alpha, beta)
        if np.random.random() < rho:
            return rate_new
        return rate

    def _precalculations(self, s, i, j, action):
        """
        Precalculate the vector s.
        """
        C = self.coupon_matrix
        s_new = np.copy(s)
        if action == 'add':
            s_new -= (C[i:i+1, :] * np.array([k > j for k in range(self.n_samples)]) + C[j:j+1, :]).reshape(-1, 1)
        elif action == 'remove':
            s_new += (C[i:i+1, :] * np.array([k > j for k in range(self.n_samples)]) + C[j:j+1, :]).reshape(-1, 1)
        return s_new

    def metropolis_within_gibbs(self, iters, hypers):
        """
        Performs Gibbs sampler with Metropolis Hastings to derive the
        conditional distributions.
        """
        G_S = self.G_R
        trace = {'graphs': [], 'rate': []}
        alpha = hypers['alpha']
        beta = hypers['beta']

        A = nx.adjacency_matrix(G_S)
        u = self.degrees - A.sum(axis=1)
        s = np.sum(np.tril(A@self.coupon_matrix), axis=0).reshape(-1, 1) + (self.coupon_matrix.T)@u
        add_GS, rem_GS = self._proposal_ratio(G_S, u)

        d = np.dot(s.flatten(), self.waiting_times.flatten())
        rate = (self.n_samples - self.n_seeds) / d[0,0]

        for _ in tqdm(range(iters)):
            G_S, s, u, add_GS, rem_GS = self.sample_graph_given_rate(G_S, rate, u, s, add_GS, rem_GS)
            rate = self.sample_rate_given_graph(rate, s, alpha, beta)
            trace['graphs'].append(G_S)
            trace['rate'].append(rate)
        return trace

if __name__ == '__main__':
    pass