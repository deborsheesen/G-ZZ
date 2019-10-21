data {
    int<lower=0> N;
    int<lower=0> d;
    int y[N];
    matrix[N,d] X;
}
parameters {
    vector[d] xi;
    vector<lower=0>[d-1] lambda2;
    real<lower=0> tau2;
    vector<lower=0>[d-1] nu;
    real<lower=0> gamma;
}
model {
    vector[N] alpha;
    
    // Hyperpriors:
    nu ~ inv_gamma(0.5, 1);
    gamma ~ inv_gamma(0.5, 1);
    
    tau2 ~ inv_gamma(0.5,1/gamma);
    for (i in 1:(d-1)){
        lambda2[i] ~ inv_gamma(0.5,1/nu[i]);
    }

    // Priors
    xi[1] ~ normal(0,1);
    for (i in 2:d){
        xi[i] ~ normal(0,sqrt(lambda2[i-1]*tau2));
    }
    alpha = X*xi;
    y ~ binomial_logit(1,alpha);
}
