functions {
  real gumbel_type2_lpdf(real tau, real lambda){
    return -(3/2 * log(tau) + lambda / sqrt(tau)); 
  }
}
data {
   int<lower = 0> n;
   real<lower = 0> lambda;
   cov_matrix[n] c; 
   vector[n] y;  
}
parameters {
   real<lower = 0> tau;
}
model {
   tau ~ gumbel_type2(lambda);
   y ~ multi_normal(rep_vector(0, n), (1/sqrt(tau)) * c);
}