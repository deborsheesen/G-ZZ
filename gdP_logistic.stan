data {
    int<lower=0> N;
    int<lower=0> d;
    int y[N];
    matrix[N,d] X;
}
parameters {
    vector[d] xi;
    vector<lower=0>[d-1] lambda;
    vector<lower=0>[d-1] tau;
    real<lower=0> sigma2;
}
model {
    vector[N] alpha;
    
    // Hyperpriors:
     for (i in 1:(d-1)){
        tau[i] ~ exponential(lambda[i]^2/2);
        lambda[i] ~ gamma(1,1);
    }
    sigma2 ~ gamma(1,1);

    // Priors
    xi[1] ~ normal(0,1);
    for (i in 2:d){
        xi[i] ~ normal(0,sqrt(sigma2*tau[i-1]));
    }
    alpha = X*xi;
    y ~ binomial_logit(1,alpha);
}
