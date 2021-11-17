#! usr/bin/env python 

import numpy as np

def generate_dataset(n_samples, precision, corr): 

    B = np.random.binomial(n=1, p=0.1, size=(n_samples, n_samples))
    B = 0.5 * (B + B.T)
    D = np.diag([sum(B[i,:]) for i in range(n)])
    cov_matrix = np.linalg.inv(D - corr * B) / precision

    dataset = np.random.multivariate_normal(mean=np.zeros(n_samples),
                                            cov=cov_matrix, 
                                            size = 10000)

    return dataset

if __name__ == '__main__':

    tau = 1
    n = 100

    for rho in [0.5, 0.8, 0.9, 0.99, 0.999]: 

        dataset = generate_dataset(n, tau, rho)
        correlations = [np.corrcoef(dataset[:,i], dataset[:,j])[0,1] for i in range(n) for j in range(n)]
    
        print("rho = {}, interval = {}".format(rho, np.quantile(correlations, q=[0.125, 0.99])))
