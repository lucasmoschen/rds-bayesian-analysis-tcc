data {
    int<lower = 0> n_pos;
    int<lower = 0> n_neg; 
    int Y_p;
    int Y_n;
    vector<lower = 0>[4] a; 
    vector<lower = 0>[4] b;
}
parameters {
    vector<lower = 0, upper = 1>[3] Z; 
    vector<lower = 0>[4] alpha;
}
transformed parameters{ 
    real<lower = 0, upper = 1> sens;
    real<lower = 0, upper = 1> spec; 
    vector<lower = 0>[3] alpha_sum;
    alpha_sum[3] = alpha[4];
    alpha_sum[2] = alpha[3] + alpha_sum[3];
    alpha_sum[1] = alpha[2] + alpha_sum[2];
    sens = 1 - Z[1] * Z[2]; //(1 - Z[1]) + Z[1] * (1 - Z[2])
    spec = 1 - Z[1] + Z[1] * Z[2] * (1 - Z[3]);
}
model {
    alpha ~ gamma(a, b);
    Z ~ beta(alpha_sum, alpha[1:3]);
    Y_p ~ binomial(n_pos, sens);
    Y_n ~ binomial(n_neg, spec);
}