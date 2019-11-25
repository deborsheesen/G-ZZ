using Distributions, TimeIt, ProgressMeter, PyPlot, CSV, JLD
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/zz_samplers.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/mbsampler.jl")

function run_sampler(max_attempts, lambda, mb_size) 

    m = CSV.read("/home/postdoc/dsen/Desktop/G-ZZ/data/pd_speech_features.csv", header=false);
    X = zeros(size(m,1)-2,size(m,2)-2)
    for i in 3:size(m,1) 
        for j in 2:size(m,2)-1 
            X[i-2,j-1] = parse(m[i,j])
        end
    end
    mu, sd = vec(mean(X[:,2:end],1)), vec(std(X[:,2:end],1))
    X_normalised = zeros(size(X))
    for i in 1:size(X,1) 
        X_normalised[i,2:end] = (X[i,2:end]-mu)./sd
    end
    X = sparse(X_normalised')
    X[1,:] = 1
    y = vec([parse(m[i,end]) for i in 3:size(m,1)])

    d, Nobs = size(X)
    σ02 = 1
    prior = SS_prior(d, σ02)
    d_hyp = hyperparam_size(prior)

    my_ll = ll_logistic_sp(X,y)
    my_model = model(my_ll, prior)

    # Sub-sampling without control variates and with weights:

    gs = Array{mbsampler}(d)
    gs[1] = umbsampler(0, Nobs, mb_size)

    for i in 2:d
        weights_het = abs.(X[i,:])./sum(abs.(X[i,:]))
        if length(X[i,:].nzind) < length(X[i,:]) 
            gs[i] = spwumbsampler(Nobs, mb_size, weights_het, prob_het)
        else 
            gs[i] = wumbsampler(Nobs, mb_size, weights_het)
        end
    end
    gs_list = mbsampler_list(d,gs);

    # 10 random projections:
    A_xi, A_hyp = load("/xtmp/GZZ_data/shrinkage_prior/speech_features/projection_matrix.jld", "A_xi", "A_hyp")

    opf = projopf(A_xi, A_hyp, 1000)
    opt = maxa_opt(max_attempts)
    outp = outputscheduler(opf,opt)
    bb = linear_bound(my_model.ll, my_model.pr, gs_list)
    mstate = zz_state(d)
    update_bound(bb, my_ll, prior, gs_list, mstate);

    adapt_speed = "by_var"
    L = 1
    my_zz_sampler = zz_sampler(0, gs_list, bb, L, adapt_speed)
    hyper_sampler = block_gibbs_sampler(lambda)
    blocksampler = Array{msampler}(2)
    blocksampler[1] = my_zz_sampler
    blocksampler[2] = hyper_sampler;

    ZZ_block_sample(my_model, outp, blocksampler, mstate)
    
    filename = "/xtmp/GZZ_data/shrinkage_prior/speech_features/lambda:"*
    string(lambda)*"-mb_size:"*string(mb_size)*".jld"

    save(filename, "xt_skeleton", outp.opf.xi_skeleton, 
        "bt_skeleton", outp.opf.bt_skeleton, 
        "hyper_skeleton", outp.opf.hyper_skeleton, 
        "alpha_skeleton", outp.opf.alpha_skeleton)
    
    outp.opf.xi_skeleton = [] 
    outp.opf.bt_skeleton = [] 
    outp.opf.hyper_skeleton = [] 
    outp.opf.alpha_skeleton = [] 
    gc()
    
end