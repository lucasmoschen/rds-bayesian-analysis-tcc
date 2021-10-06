data {
    int<lower = 0> n_pos;
    int<lower = 0> n_neg; 
    int Y_p;
    int Y_n;
    vector<lower = 0>[4] alpha_data;
}
parameters {
    simplex[4] U; 
}
transformed parameters{ 
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec; 
    sens = U[1] + U[2];
    spec = U[1] + U[3];
}
model {
    U ~ dirichlet(alpha_data);
    Y_p ~ binomial(n_pos, sens);
    Y_n ~ binomial(n_neg, spec);
}
generated quantities {
  simplex[4] U_prior;
  real<lower = 0, upper = 1> sens_prior;
  real<lower = 0, upper = 1> spec_prior;
  int Y_p_prior;
  int Y_n_prior;

  U_prior = dirichlet_rng(alpha_data);
  sens_prior = U_prior[1] + U_prior[2];
  spec_prior = U_prior[1] + U_prior[3];
  Y_p_prior = binomial_rng(n_pos, sens_prior);
  Y_n_prior = binomial_rng(n_neg, spec_prior);
}