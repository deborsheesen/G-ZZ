data 
{
    int<lower=0> nteams;
    int<lower=0> tsize;
    int<lower=0> lsize;
    int<lower=0> N;
    int teams[N, 2];
    int lineups[N, 2, lsize];
    real y[N];
    int ntrials[N];
}

parameters 
{
    matrix[nteams, tsize] beta;
    matrix [nteams, tsize] angles;
    matrix [nteams, tsize] gamma;
    real<lower=0> tau_beta;
    real<lower=0> tau_obs;
    real<lower=0> tau_angle;
    real<lower=0> lambda;
}

transformed parameters 
{
    matrix[nteams, tsize] beta_shifted;
    matrix[nteams, tsize] angles_shifted;
    beta_shifted = beta - mean(beta);
    for (i in 1:nteams)
    {
       angles_shifted[i,:] = angles[i,:] - mean(angles[i,:]);
    }
}

model 
{
    real alpha;
    real interaction;
    real totability[2];
    real beta_lineup[2,lsize];
    real angles_lineup[2,lsize];
    real gamma_lineup[2,lsize];

    // Hyperpriors:
    tau_beta ~ cauchy(0,5);
    tau_obs ~ cauchy(0,5);
    tau_angle ~ cauchy(0,5);
    lambda ~ exponential(1);

    // Priors:
    for (i in 1:nteams)
    {
        beta[i,:] ~ double_exponential(0, tau_beta);
        angles[i,:] ~ double_exponential(0, tau_angle);
    }

    // Likelihood:
    for (n in 1:N)
    {
        totability[1] = 0;
        totability[2] = 0;
        for (i in 1:2){
            for (j in 1:lsize) 
            {
                beta_lineup[i,j] = beta_shifted[teams[n,i], lineups[n,i,j]];
                angles_lineup[i,j] = angles_shifted[teams[n,i], lineups[n,i,j]];
                gamma_lineup[i,j] = gamma[teams[n,i], lineups[n,i,j]];
                totability[i] += beta_lineup[i,j];
                for (k in 1:(j-1))
                {
                    //interaction = angles_lineup[i,j]-angles_lineup[i,k];
                    interaction = gamma[i,j]*gamma[i,k];
                    totability[i] += (interaction);
                }
            }
       }
       alpha = totability[1]-totability[2];
       y[n] ~ normal(alpha, tau_obs);
    }
}



