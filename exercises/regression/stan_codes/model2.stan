data {
  int<lower = 0> positive_tests;        
  int<lower = 0> number_tests;

  int<lower = 0> J_spec;
  int<lower = 0> n_spec[J_spec];
  int<lower = 0> neg_tests_neg_subj[J_spec];

  int<lower = 0> J_sens;
  int<lower = 0> n_sens[J_sens];
  int<lower = 0> pos_tests_pos_subj[J_sens];

  // hyperparameters  
  real<lower = 0> alpha_pi;
  real<lower = 0> beta_pi;
  real mean_hyper_mean_spec;
  real mean_hyper_mean_sens;
  real<lower = 0> sd_hyper_mean_spec; 
  real<lower = 0> sd_hyper_mean_sens; 
  real<lower = 0> sd_hyper_sd_spec; 
  real<lower = 0> sd_hyper_sd_sens; 
}
parameters {
  real<lower = 0, upper = 1> pi;

  real mean_logit_spec;
  real mean_logit_sens;
  real<lower = 0> sd_logit_spec; 
  real<lower = 0> sd_logit_sens; 

  // vector[J_spec] logit_spec;
  // vector[J_sens] logit_sens;

  vector<offset=mean_logit_spec, multiplier=sd_logit_spec>[J_spec] logit_spec;
  vector<offset=mean_logit_sens, multiplier=sd_logit_sens>[J_sens] logit_sens;
}
transformed parameters {
  vector[J_spec] spec = inv_logit(logit_spec);
  vector[J_sens] sens = inv_logit(logit_sens);

  real p = (1 - spec[1])*(1 - pi) + sens[1]*pi;
}
model {
    pi   ~ beta(alpha_pi,beta_pi);

    mean_logit_spec ~ normal(mean_hyper_mean_spec, sd_hyper_mean_spec);
    sd_logit_spec ~ normal(0, sd_hyper_sd_spec);

    mean_logit_sens ~ normal(mean_hyper_mean_sens, sd_hyper_mean_sens);
    sd_logit_sens ~ normal(0, sd_hyper_sd_sens);

    logit_spec ~ normal(mean_logit_spec, sd_logit_spec);
    logit_sens ~ normal(mean_logit_sens, sd_logit_sens);

    positive_tests ~ binomial(number_tests, p);
    neg_tests_neg_subj ~ binomial(n_spec, spec); 
    pos_tests_pos_subj ~ binomial(n_sens, sens);
}
generated quantities{
    int<lower = 0> positive_tests_rep = binomial_rng(number_tests, p); 
}