data {
    int<lower = 0> n_samples;
    int<lower = 0> n_predictors; 
  
    int T[n_samples];
    matrix[n_samples, n_predictors] x;
    
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
    real<lower = 0, upper = 1> rho; 
}
transformed data{
  vector[n_samples] zeros;
  matrix<lower = 0>[n_samples, n_samples] D;
  
  for (i in 1:n_samples) {
       D[i,i] = sum(adj_matrix[i, ]);
  }
  zeros = rep_vector(0, n_samples);
}
parameters {
    vector[n_predictors] effects; 
    real<lower = 0, upper = 1> prev;
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec;
    
    vector[n_samples] omega; 
    real<lower = 0> tau; 
}
transformed parameters {
    vector[n_samples] theta;
    vector[n_samples] p; 
    
    for (i in 1:n_samples) {
        theta[i] = inv_logit(logit(prev) + x[i] * effects + omega[i]);
        p[i] = sens*theta[i] + (1 - spec)*(1 - theta[i]);
    }
}
model {
    tau ~ gamma(alpha_tau, beta_tau); 
    omega ~ multi_normal_prec(zeros, tau * (D - rho * adj_matrix));

    effects ~ multi_normal(mu, Sigma);
    prev ~ beta(alpha_p, beta_p);
    
    sens ~ beta(alpha_s, beta_s);
    spec ~ beta(alpha_e, beta_e);

    for (i in 1:n_samples) {
       T[i] ~ bernoulli(p[i]);
    }
}
generated quantities {
  vector[n_predictors] effects_prior = multi_normal_rng(mu, Sigma); 
  real<lower = 0, upper = 1> prev_prior = beta_rng(alpha_p, beta_p); 
  real<lower = 0, upper = 1> sens_prior = beta_rng(alpha_s, beta_s);
  real<lower = 0, upper = 1> spec_prior = beta_rng(alpha_e, beta_e);
}