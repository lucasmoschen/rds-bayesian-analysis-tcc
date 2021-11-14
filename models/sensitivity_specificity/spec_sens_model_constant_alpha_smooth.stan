data {
    int<lower = 0> n_pos;
    int<lower = 0> n_neg; 
    int Y_p;
    int Y_n;
    vector<lower = 0>[4] alpha_data;
}
transformed data {
   vector<lower = 0>[3] alpha_sum;
    alpha_sum[3] = alpha_data[4];
    alpha_sum[2] = alpha_data[3] + alpha_sum[3];
    alpha_sum[1] = alpha_data[2] + alpha_sum[2];
}
parameters {
    vector<lower = 0, upper = 1>[3] Z; 
}
transformed parameters{ 
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec; 
    sens = 1 - Z[1] * Z[2]; //(1 - Z[1]) + Z[1] * (1 - Z[2])
    spec = 1 - Z[1] + Z[1] * Z[2] * (1 - Z[3]);
}
model {
    Z ~ beta(alpha_sum, alpha_data[1:3]);
    Y_p ~ binomial(n_pos, sens);
    Y_n ~ binomial(n_neg, spec);
}
generated quantities {
  real<lower = 0, upper = 1> Z_prior[3];
  real<lower = 0, upper = 1> sens_prior;
  real<lower = 0, upper = 1> spec_prior;
  int Y_p_prior;
  int Y_n_prior;

  Z_prior = beta_rng(alpha_sum, alpha_data[1:3]);
  sens_prior = 1 - Z_prior[1] * Z_prior[2];
  spec_prior = 1 - Z_prior[1] + Z_prior[1] * Z_prior[2] * (1 - Z_prior[3]);
  Y_p_prior = binomial_rng(n_pos, sens_prior);
  Y_n_prior = binomial_rng(n_neg, spec_prior);
}