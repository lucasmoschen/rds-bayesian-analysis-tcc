data {
    int<lower=0> n_samples;
    int<lower=0> n_predictors; 
  
    int<lower=0, upper=1> Y[n_samples];
    matrix[n_samples, n_predictors] X;
    
    cov_matrix[n_predictors] Sigma; 
    vector[n_predictors] mu;
    real<lower=0> alpha_p; 
    real<lower=0> beta_p;
}
transformed data {
  matrix[n_predictors, n_predictors] sigma_beta;
  sigma_beta = cholesky_decompose(Sigma);
}
parameters {
    vector[n_predictors] normal_raw; 
    real<lower=0, upper=1> prev; 
}
transformed parameters {
    vector[n_predictors] effects = mu + sigma_beta * normal_raw;
}
model {
    normal_raw ~ std_normal();
    prev ~ beta(alpha_p, beta_p);
    Y ~ bernoulli_logit(logit(prev) + X * effects);
}
generated quantities {
  vector[n_predictors] effects_prior = multi_normal_rng(mu, Sigma); 
  real<lower = 0, upper = 1> prev_prior = beta_rng(alpha_p, beta_p);  
}