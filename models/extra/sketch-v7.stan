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
  real wcar_normal_lpdf(vector phi, real rho, 
                        vector A_w, int[] A_v, int[] A_u,  
                        vector Delta_inv, real log_det_Delta_inv,
                        vector lambda, int n) {
        real ztDz;  // z transpose * D * z
        real ztAz;  // z transpose * A * z
        vector[n] ldet_ImrhoC;
        ztDz = (phi .* Delta_inv)' * phi;
        ztAz = phi' * csr_matrix_times_vector(n,n,A_w, A_v, A_u, phi);
        for (i in 1:n) ldet_ImrhoC[i] = log1m(rho * lambda[i]);
        return 0.5 * (log_det_Delta_inv + sum(ldet_ImrhoC) - ztDz + rho * ztAz);

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
  matrix[n,p] X;

  matrix<lower = 0, upper = 1>[n, n] W; // adjacency matrix
  int W_n;                // number of adjacent region pairs

  real<lower = 0> alpha_tau;
  real<lower = 0> beta_tau;
  real<lower = 0> lambda_tau; 

  real<lower = 0> alpha_prev;
  real<lower = 0> beta_prev;

  vector[p] mu_beta;
  cov_matrix[p] cov_beta;
  
  int<lower = 0, upper = 1> gumbel_prior;
}
transformed data {
  matrix[p,p] sigma_beta; 
  vector[n] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n] lambda;       // eigenvalues of invsqrtD * W * invsqrtD

  vector[2*W_n] A_w; 
  int A_v[2*W_n]; 
  int A_u[n+1]; 
  

  sigma_beta = cholesky_decompose(cov_beta);
  
  for (i in 1:n) D_sparse[i] = sum(W[i]);
  {
    vector[n] invsqrtD;  
    for (i in 1:n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }

  A_w = rep_vector(1, 2 * W_n);
  A_v = csr_extract_v(W);
  A_u = csr_extract_u(W);
}
parameters {
  vector[n] phi;
  real<lower = 0> sigma;
  real<lower = 0, upper = 1> prev;
  vector[p] normal_raw; 
  real<lower = 0, upper = 1> alpha;
}
transformed parameters {
   vector[p] beta; 
   beta = mu_beta + sigma_beta * normal_raw; 
}
model {
  if (gumbel_prior == 1) {
     sigma ~ inv_gumbel_type2(lambda_tau);
  } else {
     sigma ~ gamma(alpha_tau, beta_tau);
  }
    
  phi ~ wcar_normal(alpha, A_w, A_v, A_u, D_sparse, log(prod(D_sparse)), lambda, n);
  normal_raw ~ std_normal();   
  prev ~ beta(alpha_prev, beta_prev);
  
  y ~ bernoulli_logit(logit(prev) + X * beta + sigma * phi);
}