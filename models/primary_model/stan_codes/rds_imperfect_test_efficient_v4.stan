data {
    int<lower = 0> n_samples;
  
    int T[n_samples];

    real<lower = 0> alpha_p; 
    real<lower = 0> beta_p;
    real<lower = 0> alpha_tau; 
    real<lower = 0> beta_tau;
    
    matrix[n_samples, n_samples] cov_cholesky; 
    real<lower = 0, upper = 1> rho; 
}
parameters {
    real<lower = 0, upper = 1> prev;

    vector[n_samples] omega_raw;
    real<lower = 0> tau; 
}
transformed parameters {
}
model {
    tau ~ gamma(alpha_tau, beta_tau); 
    omega_raw ~ std_normal();
    prev ~ beta(alpha_p, beta_p);

    T ~ bernoulli_logit(logit(prev) + (1/sqrt(tau)) * cov_cholesky * omega_raw);
}