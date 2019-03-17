data {
    int<lower=0> N;
    int<lower=0> d;
    int y[N];
    matrix[N,d] X;
}
parameters {
    vector[d] xi;
    vector<lower=0>[d-1] Psi;
    real<lower=0> lambda;
    real<lower=0> lambdagamma2;
}
transformed parameters {
    real<lower=0> gamma2;
    gamma2 = lambdagamma2/(2*lambda);
}
model {
    vector[N] alpha;
    real M;
    
    M = 1;

    // Hyperpriors:
    Psi ~ gamma(lambda, 1/(2*gamma2));
    lambda ~ exponential(1);
    lambdagamma2 ~ inv_gamma(2,M);

    // Priors
    xi[1] ~ normal(0,1);
    for (i in 2:d){
        xi[i] ~ normal(0,Psi[i-1]);
    }
    alpha = X*xi;
    y ~ binomial_logit(1,alpha);
}
