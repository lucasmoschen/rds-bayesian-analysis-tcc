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
    
    matrix[n_samples, n_samples] inv_precision; 
}
transformed data {
   matrix[n_predictors, n_predictors] sigma;
   matrix[n_predictors, n_predictors] cov_cholesky;
   sigma = cholesky_decompose(Sigma);
   cov_cholesky = cholesky_decompose(inv_precision);

}
parameters {
    vector[n_predictors] effects_raw; 
    vector[n_samples] omega_raw;

    real<lower = 0, upper = 1> prev;
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec;
    real<lower = 0> tau;
}
transformed parameters {
    vector[n_predictors] effects;
    vector[n_samples] omega;
    vector[n_samples]<lower = 0, upper = 1> p;
    effects = mu + sigma * effects_raw;
    omega = (1/sqrt(tau)) * cov_cholesky * omega_raw;
    p = (1 - spec) + (sens + spec - 1) * inv_logit(logit(prev) + X * effects + omega);
}
model {
    tau ~ gamma(alpha_tau, beta_tau); 

    omega_raw ~ std_normal();
    effects_raw ~ std_normal();

    prev ~ beta(alpha_p, beta_p);
    sens ~ beta(alpha_s, beta_s);
    spec ~ beta(alpha_e, beta_e);

    Y ~ bernoulli(p);
}
generated quantities {
   vector[n_samples] theta;
   theta = inv_logit(logit(prev) + X * effects + omega);
}
