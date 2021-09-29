data {
    int n_samples; 
    vector[n_samples] y; 
    vector[n_samples] sigma;
}
parameters {
   real mu; 
   real<lower = 0> tau; 
   vector[n_samples] theta; 
}
model {
   mu ~ normal(0, 10);
   tau ~ cauchy(0, 10);

   for (n in 1:n_samples) {
      theta[n] ~ normal(mu, tau); 
      y[n] ~ normal(theta[n], sigma[n]);
   }
}