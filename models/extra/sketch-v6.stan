functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (sum(ldet_terms) - (phit_D * phi - alpha * (phit_W * phi)));
  }
  real gumbel_type2_lpdf(real tau, real lambda){
    return -(3.0/2.0 * log(tau) + lambda / sqrt(tau)); 
  }
  real inv_gumbel_type2_lpdf(real sigma, real lambda){
    return -lambda * sigma;
  }
}
data {
  int<lower = 1> n;
  int<lower = 1> p;
  int<lower = 0, upper = 1> y[n];
  matrix[n,p+1] X;

  matrix<lower = 0, upper = 1>[n, n] W; // adjacency matrix
  int W_n;                // number of adjacent region pairs

  real<lower = 0> alpha_tau;
  real<lower = 0> beta_tau;
  real<lower = 0> lambda_tau; 

  vector[p+1] mu_beta;
  cov_matrix[p+1] cov_beta;
  
  int<lower = 0, upper = 1> gumbel_prior;
}
transformed data {
  matrix[p+1,p+1] sigma_beta; 
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[n] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n] lambda;       // eigenvalues of invsqrtD * W * invsqrtD

  sigma_beta = cholesky_decompose(cov_beta);
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:n) D_sparse[i] = sum(W[i]);
  {
    vector[n] invsqrtD;  
    for (i in 1:n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
parameters {
  vector[n] phi;
  real<lower = 0> sigma;
  real<lower = 0, upper = 1> prev;
  vector[p+1] normal_raw; 
  real<lower = 0, upper = 1> alpha;
}
transformed parameters {
   vector[p+1] beta; 
   beta = mu_beta + sigma_beta * normal_raw; 
}
model {
  if (gumbel_prior == 1) {
     sigma ~ inv_gumbel_type2(lambda_tau);
  } else {
     sigma ~ gamma(alpha_tau, beta_tau);
  }
  
  phi ~ sparse_car(alpha, W_sparse, D_sparse, lambda, n, W_n);
  normal_raw ~ std_normal();   
  
  y ~ bernoulli_logit(X * beta + sigma * phi);
}