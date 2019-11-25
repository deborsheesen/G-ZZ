using Distributions, TimeIt, ProgressMeter, PyPlot, CSV, JLD
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/zz_samplers.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/mbsampler.jl")

function run_sampler(max_attempts, lambda, mb_size, adapt_speed="by_var") 

    # Read data:
    m = CSV.read("/home/postdoc/dsen/Desktop/G-ZZ/data/gastroenterology.txt", header=false)
    convert(Array,m[2,:])

    # Get data into the right format:
    x = zeros(size(m,1)-3,size(m,2))
    for i in 4:size(m,1) 
        for j in 1:size(m,2)
            x[i-3,j] = parse(Float64, m[i,j])
        end
    end
    x = x[vec(sum(abs.(x),2) .!= 0),:]

    X = spzeros(size(x,1)+1,size(x,2))
    mu, sd = vec(mean(x,2)), vec(std(x,2))
    for i in 1:size(x,2)
        X[2:end,i] = (x[:,i]-mu)./sd
    end
    X[1,:] = 1

    y = zeros(size(m,2))
    for j in 1:size(m,2) 
        label = parse(Int64, m[2,j])
        y[j] = (label>1)
    end
    
    # Define model:
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

    A_xi, A_hyp = zeros(50,d), zeros(50,d_hyp)
    for i in 1:size(A_xi,1)
        A_xi[i,8*i], A_hyp[i,8*i] = 1, 1 
    end

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
    
    filename = "/xtmp/GZZ_data/shrinkage_prior/gastroenterology/lambda:"*
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