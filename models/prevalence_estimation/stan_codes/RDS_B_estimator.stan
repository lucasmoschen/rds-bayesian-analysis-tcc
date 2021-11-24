functions {
   real log_bernoulli_logit(int y, real alpha){
       return alpha * (y-1) - log1p(exp(-alpha));
   }
}
data {
    int<lower=0> n_samples;  
    int<lower=0, upper=1> Y[n_samples];
    real mu_alpha; 
    real<lower=0> sigma_alpha;
    vector[n_samples] delta;
}
parameters {
    real alpha; 
}
model {
    for (i in 1:n_samples) {
       target += delta[i] * log_bernoulli_logit(Y[i], alpha);
    }
    target += -0.5 * (alpha - mu_alpha) * (alpha - mu_alpha) / sigma_alpha * sigma_alpha; 
}
generated quantities {
  real<lower=0, upper=1> theta;
  theta = inv_logit(alpha);
}