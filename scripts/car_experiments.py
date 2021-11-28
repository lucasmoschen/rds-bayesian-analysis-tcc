#! usr/bin/env python
"""
Experiments for understansing true correlation from different rho specifications.
"""

import numpy as np
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def generate_dataset(n_samples, precision, corr):
    """
    Generate matrix of connections and dataset with 10000 CARs.
    """
    B_adj = np.random.binomial(n=1, p=1/n_samples, size=(n_samples, n_samples))
    np.fill_diagonal(B_adj, 0)
    for i in range(n_samples):
        if B_adj[i, :].sum() == 0:
            j = np.random.randint(n-1)
            if j < i: B_adj[i, j] = 1
            else: B_adj[i, j+1] = 1
    B_adj = 0.5 * (B_adj + B_adj.T)
    D = np.diag([B_adj[i, :].sum() for i in range(n_samples)])
    cov_matrix = np.linalg.inv(D - corr * B_adj) / precision

    dataset = np.random.multivariate_normal(mean=np.zeros(n_samples),
                                            cov=cov_matrix,
                                            size=10000)

    return dataset, B_adj

@jit(nopython=True, parallel=True)
def calculating_pearson_correlations(data, n_samples):
    means = [data[:, i].mean() for i in range(n_samples)]
    standard_deviations = [data[:, i].std() for i in range(n_samples)]

    def correlation(data, ind_x, ind_y, means, standard_deviations, m):
        covariance = sum((data[:, ind_x] - means[ind_x]) * (data[:, ind_y] - means[ind_y]))
        return covariance/(m * standard_deviations[ind_x] * standard_deviations[ind_y])
        
    m = data.shape[0]
    correlations = [correlation(data, i, j, means, standard_deviations, m) for i in range(n_samples) for j in range(i, n_samples)]
    return correlations

@jit(nopython=True, parallel=True)
def calculating_moran_correlations(data, n, M):

    moran_values = np.zeros(10000)
    M_sum = M.sum()
    for k in range(10000):
        centered = data[k, :] - data[k, :].mean()
        moran = sum([2 * M[i, j] * centered[i] * centered[j]
                     for i in range(n) for j in range(i+1, n)])
        moran *= 1/(M_sum*data[k].var())
        moran_values[k] = moran
    return moran_values

if __name__ == '__main__':

    tau = 1
    n = 500
    array_moran = np.zeros((75, 5))
    array_pearson = np.zeros((75, 5))
    rho_values = np.zeros(75)
    rho_values[:45] = np.linspace(0, 0.9, 45, endpoint=False)
    rho_values[45:65] = np.linspace(0.9, 1, 20, endpoint=False)
    rho_values[65:] = np.linspace(0.995, 1, 11, endpoint=False)[1:]

    exp = 0
    if exp:
        for index__, rho in tqdm(enumerate(rho_values)):
            dataset, B_adj = generate_dataset(n, tau, rho)
            correlations = calculating_moran_correlations(dataset, n, B_adj)
            correlations2 = calculating_pearson_correlations(dataset, n)
            array_moran[index__, :] = np.quantile(correlations, q=[0.025, 0.125, 0.5, 0.875, 0.975])
            array_pearson[index__, :] = np.quantile(correlations2, q=[0.025, 0.125, 0.5, 0.875, 0.975])
        np.save('/home/lucasmoschen/Documents/GitHub/rds-bayesian-analysis/data/experiments/rho_moran.npy', array_moran)
        np.save('/home/lucasmoschen/Documents/GitHub/rds-bayesian-analysis/data/experiments/rho_pearson.npy', array_pearson)

    else:
        array_moran = np.load('/home/lucasmoschen/Documents/GitHub/rds-bayesian-analysis/data/experiments/rho_moran.npy')
        array_pearson = np.load('/home/lucasmoschen/Documents/GitHub/rds-bayesian-analysis/data/experiments/rho_pearson.npy')
        #plt.plot(rho_values, array_moran[:,2])
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].fill_between(rho_values,
                           y1=array_moran[:, 0],
                           y2=array_moran[:, 4],
                           alpha=0.3,
                           color='midnightblue',
                           label='HDI 95%'
                           )
        ax[0].fill_between(rho_values,
                           y1=array_moran[:, 1],
                           y2=array_moran[:, 3],
                           alpha=0.3,
                           color='darkred',
                           label='HDI 75%'
                           )
        ax[0].plot(rho_values, array_moran[:, 2], color='black', label='Median')
        ax[0].axhline(0, linestyle='--', color='red')
        ax[1].axhline(0, linestyle='--', color='red')
        ax[0].set_xlabel(r'$\rho$', fontsize=16)
        ax[0].set_ylabel("Moran's I autocorrelation", fontsize=16)
        ax[0].legend(loc='upper left')
        ax[1].fill_between(rho_values,
                           y1=array_pearson[:, 0],
                           y2=array_pearson[:, 4],
                           alpha=0.3,
                           color='midnightblue',
                           label='HDI 95%'
                           )
        ax[1].fill_between(rho_values,
                           y1=array_pearson[:, 1],
                           y2=array_pearson[:, 3],
                           alpha=0.3,
                           color='darkred',
                           label='HDI 75%'
                           )
        ax[1].plot(rho_values, array_pearson[:, 2], color='black', label='Median')
        #ax[1].set_xscale('log')
        ax[1].set_xlabel(r'$\rho$', fontsize=16)
        ax[1].set_ylabel("Pearson's correlation", fontsize=16)
        ax[1].legend(loc='upper left')
        ax[1].set_xlim((0.9, 1))
        plt.savefig('/home/lucasmoschen/Documents/GitHub/rds-bayesian-analysis/images/correlation-different-rho-values-car.pdf', bbox_inches='tight')
        plt.show()

    rho = 0.95
    array_moran = np.zeros((5, 2))
    array_pearson = np.zeros((5, 2))
    for index__, n in tqdm(enumerate([10, 50, 100, 500, 1000])):
        dataset, B_adj = generate_dataset(n, tau, rho)
        correlations = calculating_moran_correlations(dataset, n, B_adj)
        correlations2 = calculating_pearson_correlations(dataset, n)
        array_moran[index__, :] = np.quantile(correlations, q=[0.125, 0.875])
        array_pearson[index__, :] = np.quantile(correlations2, q=[0.125, 0.875])
        print('n = {}: Moran {} and Pearson {}'.format(n, array_moran, array_pearson))


    
