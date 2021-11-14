data {
    int<lower = 0> n_pos;
    int<lower = 0> n_neg; 
    int Y_p;
    int Y_n;
    vector[2] mu_gamma;
    cov_matrix[2] Sigma_gamma; 
}
transformed data {
   matrix[2,2] sigma_gamma;
   sigma_gamma = cholesky_decompose(Sigma_gamma);
}
parameters {
    vector[2] normal_raw;
}
transformed parameters {
   vector[2] logit_sens_spec;
   logit_sens_spec = mu_gamma + sigma_gamma * normal_raw;
}
model {
    normal_raw ~ std_normal();
    Y_p ~ binomial_logit(n_pos, logit_sens_spec[1]);
    Y_n ~ binomial_logit(n_neg, logit_sens_spec[2]);
}
generated quantities {
   real<lower = 0, upper = 1> sens;
   real<lower = 0, upper = 1> spec; 
   sens = inv_logit(logit_sens_spec[1]);
   spec = inv_logit(logit_sens_spec[2]);
}