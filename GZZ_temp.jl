# stuff for running experiments:
# Code in one compacft file 

using Distributions, TimeIt, ProgressMeter, PyPlot, JLD
include("zz_structures.jl")
include("mbsampler.jl")


#To generate data::
"""
d, Nobs = 250, 500
pξ = 1e-1
ξ_true = zeros(d)
ξ_true[1] = rand(Normal())
for i in 2:d 
    u, v = rand(), rand()
    if u < pξ 
        ξ_true[i] = rand(Uniform(5,10))*(2*(v<0.5)-1)
    end
end
ξ_true[ξ_true .!= 0] -= mean(ξ_true[ξ_true .!= 0]);

pX = 1e-1
X = rand(Normal(), d, Nobs) .* rand(Binomial(1,pX), d, Nobs)
X[1,:] = ones(Nobs)
y = [rand(Binomial(1, 1/(1+exp(-ξ_true'X[:,j]))), 1)[1] + 0. for j in 1:Nobs];
save("GZZ_data5.jld", "X", X, "y", y, "xi_true", ξ_true)
"""

function run_shrpr(dat, mb_size, max_attempts, hyp_λ, n_samples, shrpr, maxlag=100) 
    X = load(dat, "X")
    y = load(dat, "y")
    ξ_true = load(dat, "xi_true");

    d, Nobs = size(X)
    σ02 = 1
    if shrpr == "HS"
        prior = HS_prior(d, σ02)
    elseif shrpr == "GDP" 
        prior = GDP_prior(d, σ02)
    else 
        print("Error, prior type not recognised \n")
    end
            
    my_ll = ll_logistic(X,y);
    my_model = model(my_ll, prior)
    root = find_root(my_model, rand(d))

    # Sub-Sampling without control variate and with weights
    ϵ = 1e-2
    weights = abs.(X) + ϵ
    weights ./= sum(weights,2)
    gs = [wumbsampler(Nobs, mb_size, weights[i,:]) for i in 1:d]
    gs_list = mbsampler_list(d,gs);

    # Sub-Sampling with control variate and with weights
    #ϵ = 1e-2
    #weights_cv = zeros(d, Nobs)
    #for n in 1:Nobs
        #weights_cv[:,n] = [abs.(X[i,n])*norm(X[:,n]) for i in 1:d] + ϵ
    #end
    #weights_cv ./= sum(weights_cv,2);mbs = [wumbsampler(Nobs, mb_size, weights_cv[i,:]) for i in 1:d];
    #gs_list = cvmbsampler_list(my_model, mbs, root);

    A = eye(d)

    opf = projopf(A, 1000, hyperparam_size(prior))
    opt = maxa_opt(max_attempts)
    outp = outputscheduler(opf,opt)
    bb = linear_bound(my_model.ll, my_model.pr, gs_list)
    update_bound(bb, my_ll, prior, gs_list, zz_state(opf));

    adapt_speed = false
    L = 1
    my_zz_sampler = zz_sampler(0, gs_list, bb, L, adapt_speed)
    hyper_sampler = block_gibbs_sampler(hyp_λ)
    blocksampler = Array{msampler}(2)
    blocksampler[1] = my_zz_sampler
    blocksampler[2] = hyper_sampler;


    ZZ_block_sample(my_model::model, outp::outputscheduler, blocksampler::Array{msampler});

    discard = 0
    n_samples = 10^4
    xi_samples = extract_samples(outp.opf.xi_skeleton[:,discard+1:end], 
                                 outp.opf.bt_skeleton[:,discard+1:end], 
                                 outp.opf.bt_skeleton[end]/n_samples,
                                 "linear")[:,1:end-1];
    outp = nothing 
    gc()

    maxlag = 100
    acfs = zeros(d, maxlag)
    for i in 1:d 
        acfs[i,:] = acf(xi_samples[i,:], maxlag)
    end

    cover = zeros(d)
    ci = zeros(d,2)
    for i in 1:d 
        ci[i,:] = percentile(xi_samples[i,:], [5, 95])
        cover[i] = (ci[i,1]<ξ_true[i])&(ξ_true[i]<ci[i,2])
    end
    gc()
    return xi_samples, ξ_true, acfs, cover, ci
end
