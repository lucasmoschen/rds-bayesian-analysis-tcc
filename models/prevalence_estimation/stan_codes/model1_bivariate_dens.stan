functions {
    real integrand(real x, real xc, real[] theta, real[] x_r, int[] x_i){
        real s = x_r[1];
        real t = x_r[2];
        real alpha11 = theta[1];
        real alpha10 = theta[2];
        real alpha01 = theta[3];
        real alpha00 = theta[4];
        
        return x^(alpha11-1)*(s-x)^(alpha10-1)*(t-x)^(alpha01-1)*(1-s-t+x)^(alpha00-1);
    }
    real beta_bivariate_lpdf(vector[2] x, vector[4] theta){
        real c = lgamma(sum(theta));
        for (i in 1:4) {
          c += -lgamma(theta[i]);
        }
        real lb = max(0, x[1]+x[2]-1);
        real ub = min(x[1], x[2]);
        return c + log(integrate_1d(integrand, lb, ub, alpha, x));
    }
    
}
data {
  int<lower = 0> positive_tests;        
  int<lower = 0> number_tests;
  int<lower = 0> neg_tests_neg_subj;
  int<lower = 0> n_spec;
  int<lower = 0> pos_tests_pos_subj;
  int<lower = 0> n_sens;

  real<lower = 0> alpha_pi;
  real<lower = 0> beta_pi;
  
  vector<lower = 0>[4] alpha; 
}
parameters {
  vector<lower = 0, upper = 1>[4] U;
  real<lower = 0, upper = 1> pi;
}
transformed parameters {
  real<lower = 0, upper = 1> spec = U[1] + U[2];
  real<lower = 0, upper = 1> sens = U[1] + U[3];

  real<lower = 0, upper = 1> p = (1 - spec)*(1 - pi) + sens*pi;
}
model {
  U  ~ dirichlet(alpha); 
  pi ~ beta(alpha_pi,beta_pi);
  
  real<lower = 0, upper = 1> spec = U[1] + U[2];
  real<lower = 0, upper = 1> sens = U[1] + U[3];

  real<lower = 0, upper = 1> p = (1 - spec)*(1 - pi) + sens*pi;

  positive_tests     ~ binomial(number_tests, p);
  neg_tests_neg_subj ~ binomial(n_spec, spec); 
  pos_tests_pos_subj ~ binomial(n_sens, sens);
}
//generated quantities {
//  vector<lower = 0, upper = 1>[4] U_prior = dirichlet_rng(alpha); 
//
//  real<lower = 0, upper = 1> spec_prior = U_prior[1] + U_prior[2]; 
//  real<lower = 0, upper = 1> sens_prior = U_prior[1] + U_prior[3];
// real<lower = 0, upper = 1> pi_prior   = beta_rng(alpha_pi, beta_pi);
//  
//  real p_prior = (1 - spec_prior)*(1 - pi_prior) + sens_prior*pi_prior;
//}