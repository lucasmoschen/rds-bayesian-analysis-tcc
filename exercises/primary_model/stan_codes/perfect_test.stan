data {
    int<lower = 0> n_samples;
    int<lower = 0> n_predictors; 
  
    int T[n_samples];
    matrix[n_samples, n_predictors+1] x;
    
    cov_matrix[n_predictors+1] Sigma; 
    vector[n_predictors+1] mu;
    real<lower = 0> alpha_p; 
    real<lower = 0> beta_p;
}
parameters {
    vector[n_predictors+1] effects; 
    real<lower = 0, upper = 1> prev; 
}
model {
    effects ~ multi_normal(mu, Sigma);
    prev ~ beta(alpha_p, beta_p);

    for (i in 1:n_samples) {
       T[i] ~ bernoulli_logit(logit(prev) + x[i] * effects);
    }
}
generated quantities {
  vector[n_predictors+1] effects_prior = multi_normal_rng(mu, Sigma); 
  real<lower = 0, upper = 1> prev_prior = beta_rng(alpha_p, beta_p);  
}