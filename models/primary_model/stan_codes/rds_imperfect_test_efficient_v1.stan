data {
    int<lower = 0> n_samples;
    int adj_pairs;
    int<lower=1, upper=n_samples> node1[adj_pairs];  // node1[i] adjacent to node2[i]
    int<lower=1, upper=n_samples> node2[adj_pairs];  // and node1[i] < node2[i]
  
    int T[n_samples];
    
    real<lower = 0> alpha_p; 
    real<lower = 0> beta_p;
    real<lower = 0> alpha_tau;
    real<lower = 0> beta_tau;   
}
parameters {
    real<lower = 0, upper = 1> prev;
    vector[n_samples] omega; 
    real<lower = 0> tau; 
}
transformed parameters {
    vector[n_samples] theta;
    real<lower=0> sigma_omega;  
    sigma_omega = inv(sqrt(tau));
    for (i in 1:n_samples) {
       theta[i] = inv_logit(logit(prev) + sigma_omega * omega[i]);
    }
}
model {
    tau ~ gamma(alpha_tau, beta_tau);

    target += -0.5 * dot_self(omega[node1] - omega[node2]);
    sum(omega) ~ normal(0, 0.001 * n_samples);
    
    prev ~ beta(alpha_p, beta_p);

    for (i in 1:n_samples) {
       T[i] ~ bernoulli(theta[i]);
    }
}