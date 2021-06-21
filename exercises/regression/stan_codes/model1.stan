data {
  int<lower = 0> positive_tests;        
  int<lower = 0> number_tests;
  int<lower = 0> neg_tests_neg_subj;
  int<lower = 0> n_spec;
  int<lower = 0> pos_tests_pos_subj;
  int<lower = 0> n_sens;

  real<lower = 0> alpha_spec;
  real<lower = 0> beta_spec;
  real<lower = 0> alpha_sens;
  real<lower = 0> beta_sens;
  real<lower = 0> alpha_pi;
  real<lower = 0> beta_pi;
}
parameters {
  real<lower = 0, upper = 1> spec; 
  real<lower = 0, upper = 1> sens;
  real<lower = 0, upper = 1> pi;
}
transformed parameters {
  real p = (1 - spec)*(1 - pi) + sens*pi;
}
model {
  spec ~ beta(alpha_spec,beta_spec);
  sens ~ beta(alpha_sens,beta_sens);
  pi   ~ beta(alpha_pi,beta_pi);

  positive_tests     ~ binomial(number_tests, p);
  neg_tests_neg_subj ~ binomial(n_spec, spec); 
  pos_tests_pos_subj ~ binomial(n_sens, sens);
}
generated quantities {
  real<lower = 0, upper = 1> spec_prior = beta_rng(alpha_spec, beta_spec); 
  real<lower = 0, upper = 1> sens_prior = beta_rng(alpha_sens, beta_sens);
  real<lower = 0, upper = 1> pi_prior   = beta_rng(alpha_pi, beta_pi
  
  real p_prior = (1 - spec)*(1 - pi) + sens*pi;
}