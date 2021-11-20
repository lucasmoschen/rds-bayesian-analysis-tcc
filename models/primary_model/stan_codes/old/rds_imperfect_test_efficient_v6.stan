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
  real inv_gumbel_type2_lpdf(real sigma, real lambda){
    return -lambda * sigma;
  }
} 
data {
    int<lower = 0> n_samples;
    int<lower = 0> p;
    int T[n_samples];
    matrix[n_samples,p] X;

    
    real<lower = 0> alpha_prev; 
    real<lower = 0> beta_prev;

    real<lower = 0> alpha_s; 
    real<lower = 0> beta_s;
    real<lower = 0> alpha_e; 
    real<lower = 0> beta_e;

    real<lower = 0> alpha_tau;
    real<lower = 0> beta_tau;
    real<lower = 0> M_tau;
    real<lower = 0> lambda_tau; 
    
    matrix<lower = 0, upper = 1>[n_samples, n_samples] adj_matrix; 
    int adj_pairs;

    vector[p] mu_beta;
    cov_matrix[p] cov_beta;

    int<lower = 0, upper = 2> tau_prior;     
}
transformed data{
  int adj_sparse[adj_pairs, 2];   // adjacency pairs
  vector[n_samples] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n_samples] lambda;       // eigenvalues of invsqrtD * A * invsqrtD
  matrix[p,p] sigma_beta;
  real<lower = 0, upper = 1> lambda_max; 

  sigma_beta = cholesky_decompose(cov_beta);
  
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

  lambda_max = 1/max(lambda);
}
parameters {
    real<lower = 0, upper = 1> prev;
    vector[n_samples] omega; 
    real<lower = 0, upper = M_tau> tau;
    real<lower = 0, upper = lambda_max> rho;
    vector[p] normal_raw; 

    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec;
}
transformed parameters {
   vector[p] beta; 
   real<lower = 0> sigma;
   vector<lower = 0, upper = 1>[n_samples] apparent_prev;
   vector<lower = 0, upper = 1>[n_samples] theta;

   sigma = 1/sqrt(tau);
   beta = mu_beta + sigma_beta * normal_raw; 

   theta = inv_logit(logit(prev) + sigma * omega + X * beta);
   apparent_prev = theta * sens + (1 - theta) * (1 - spec);
}
model {
    if (tau_prior == 1) {
       tau ~ gumbel_type2(lambda_tau);
    } else if (tau_prior == 2){
       tau ~ uniform(0, M_tau);
    } else {
       tau ~ gamma(alpha_tau, beta_tau);
    }

    sens ~ beta(alpha_s, beta_s);
    spec ~ beta(alpha_e, beta_e);

    rho ~ uniform(0, lambda_max);
    
    omega ~ sparse_car(rho, adj_sparse, D_sparse, lambda, n_samples, adj_pairs);
    prev ~ beta(alpha_prev, beta_prev);

    normal_raw ~ std_normal();

    T ~ bernoulli_logit(apparent_prev);
}