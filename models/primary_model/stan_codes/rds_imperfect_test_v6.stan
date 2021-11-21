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
      return 0.5 * (sum(ldet_terms) - omegat_D * omega + rho * (omegat_W * omega));
  }
} 
data {
    int<lower=0> n_samples;
    int<lower=0> n_predictors; 
  
    int<lower=0, upper=1> Y[n_samples];
    matrix[n_samples, n_predictors] X;

    cov_matrix[n_predictors] Sigma; 
    vector[n_predictors] mu;
    real<lower = 0> alpha_p; 
    real<lower = 0> beta_p;
    real<lower = 0> alpha_s; 
    real<lower = 0> beta_s;
    real<lower = 0> alpha_e; 
    real<lower = 0> beta_e;

    real<lower = 0> alpha_tau; 
    real<lower = 0> beta_tau;
    
    matrix<lower = 0, upper = 1>[n_samples, n_samples] adj_matrix; 
    int adj_pairs;
}
transformed data{
  int adj_sparse[adj_pairs, 2];   // adjacency pairs
  vector[n_samples] D_sparse;     // diagonal of D
  vector[n_samples] lambda;       // eigenvalues of invsqrtD * A * invsqrtD
  matrix[n_predictors, n_predictors] sigma;
  real max_lambda;
  real lower_bound_tau;
  
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
    max_lambda = max(lambda);
  }
  sigma = cholesky_decompose(Sigma);
}
parameters {
    vector[n_predictors] normal_raw; 
    real<lower = 0, upper = 1> prev;
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec;
    real<lower = 0, upper = 1/max_lambda> rho;
    
    vector[n_samples] omega; 
    real<lower = 0> tau; 
}
transformed parameters {
    vector<lower = 0, upper = 1>[n_samples] p;
    vector[n_predictors] effects; 
    effects = mu + sigma * normal_raw; 
    p = (1 - spec) + (spec + sens - 1) * inv_logit(logit(prev) + X * effects + (1/sqrt(tau)) * omega);
}
model {
    tau ~ gamma(alpha_tau, beta_tau);     
    rho ~ uniform(0, 1/max_lambda);
    omega ~ sparse_car(rho, adj_sparse, D_sparse, lambda, n_samples, adj_pairs);

    normal_raw ~ std_normal();
    prev ~ beta(alpha_p, beta_p);
    sens ~ beta(alpha_s, beta_s);
    spec ~ beta(alpha_e, beta_e);

    Y ~ bernoulli(p);
}
generated quantities {
   vector[n_samples] theta;
   theta = inv_logit(logit(prev) + X * effects + (1/sqrt(tau)) * omega);
}