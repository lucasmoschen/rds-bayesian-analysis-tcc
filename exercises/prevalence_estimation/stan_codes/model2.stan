functions { 
    real truncated_normal_rng(real mu, real sigma) {
          real p_lb = normal_cdf(0, mu, sigma);
          real u = uniform_rng(p_lb, 1);
          real y = mu + sigma * Phi(u);
          return y;
    }
}
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
transformed data{
    vector[J_spec] one_spec = rep_vector(1, J_spec);
    vector[J_sens] one_sens = rep_vector(1, J_sens);
}
parameters {
  real<lower = 0, upper = 1> pi;

  real mean_logit_spec;
  real mean_logit_sens;
  real<lower = 0> sd_logit_spec; 
  real<lower = 0> sd_logit_sens; 
  
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

    // predictive distribution 
    int<lower = 0> positive_tests_rep = binomial_rng(number_tests, p); 
    
    // prior predictive distribution 
    
    real<lower = 0, upper = 1> pi_prior = beta_rng(alpha_pi,beta_pi);

    real mean_logit_spec_prior = normal_rng(mean_hyper_mean_spec, sd_hyper_mean_spec);
    real mean_logit_sens_prior = normal_rng(mean_hyper_mean_sens, sd_hyper_mean_sens);

    real<lower = 0> sd_logit_spec_prior = truncated_normal_rng(0, sd_hyper_sd_spec); 
    real<lower = 0> sd_logit_sens_prior = truncated_normal_rng(0, sd_hyper_sd_sens); 
  
    real logit_spec_prior[J_spec] = normal_rng(one_spec * mean_logit_spec_prior, sd_logit_spec_prior);
    real logit_sens_prior[J_sens] = normal_rng(one_sens * mean_logit_sens_prior, sd_logit_sens_prior);
    
    real spec_prior[J_spec] = inv_logit(logit_spec_prior);
    real sens_prior[J_sens] = inv_logit(logit_sens_prior);

    real p_prior = (1 - spec_prior[1])*(1 - pi_prior) + sens_prior[1]*pi_prior;
}