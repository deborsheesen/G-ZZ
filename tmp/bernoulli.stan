data { 
    int<lower=0> N; 
    int<lower=0,upper=1> y[N];
} 
parameters {
    real<lower=0,upper=1> theta1;
    real<lower=0,upper=1> theta2;
} 
model {
    theta1 ~ beta(1,1);
    theta2 ~ beta(1,1);
    y ~ bernoulli(theta1^2+theta2);
}