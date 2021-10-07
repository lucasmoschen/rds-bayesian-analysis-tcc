functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param omega Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param rho Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of omega (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector omega, real rho, 
                       int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] omegat_D; // omega' * D
      row_vector[n] omegat_W; // omega' * W
      vector[n] ldet_terms;
    
      omegat_D = (omega .* D_sparse)';
      omegat_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        omegat_W[W_sparse[i, 1]] = omegat_W[W_sparse[i, 1]] + omega[W_sparse[i, 2]];
        omegat_W[W_sparse[i, 2]] = omegat_W[W_sparse[i, 2]] + omega[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(rho * lambda[i]);
      return 0.5 * (sum(ldet_terms) - (omegat_D * omega - rho * (omegat_W * omega)));
  }
  real gumbel_type2_lpdf(real tau, real lambda){
    return -(3.0/2.0 * log(tau) + lambda / sqrt(tau)); 
  }
} 
data {
    int<lower = 0> n_samples;
  
    int T[n_samples];
    
    real<lower = 0> alpha_p; 
    real<lower = 0> beta_p;
    real<lower = 0> alpha_tau;
    real<lower = 0> beta_tau;
    real<lower = 0> lambda_tau; 
    
    matrix<lower = 0, upper = 1>[n_samples, n_samples] adj_matrix; 
    int adj_pairs;
    real<lower = 0, upper = 1> rho;

    int<lower = 0, upper = 1> gumbel_prior;     
}
transformed data{
  int adj_sparse[adj_pairs, 2];   // adjacency pairs
  vector[n_samples] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n_samples] lambda;       // eigenvalues of invsqrtD * A * invsqrtD
  
  { // generate sparse representation for A
  int counter;
  counter = 1;
  // loop over upper triangular part of A to identify neighbor pairs
    for (i in 1:(n_samples - 1)) {
      for (j in (i + 1):n_samples) {
        if (adj_matrix[i, j] == 1) {
          adj_sparse[counter, 1] = i;
          adj_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:n_samples) D_sparse[i] = sum(adj_matrix[i]);
  {
    vector[n_samples] invsqrtD;  
    for (i in 1:n_samples) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(adj_matrix, diag_matrix(invsqrtD)));
  }
}
parameters {
    real<lower = 0, upper = 1> prev;
    vector[n_samples] omega; 
    real<lower = 0> tau; 
}
transformed parameters {

}
model {
    if (gumbel_prior == 1) {
       tau ~ gumbel_type2(lambda_tau);
    } else {
       tau ~ gamma(alpha_tau, beta_tau);
    }
    
    omega ~ sparse_car(rho, adj_sparse, D_sparse, lambda, n_samples, adj_pairs);
    prev ~ beta(alpha_p, beta_p);

    T ~ bernoulli_logit(logit(prev) + (1/sqrt(tau)) * omega);
}
generated quantities {
   vector[n_samples] theta; 
   theta = inv_logit(logit(prev) + (1/sqrt(tau)) * omega);
}