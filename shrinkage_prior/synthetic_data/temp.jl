using Distributions, TimeIt, ProgressMeter, PyPlot, JLD
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/zz_samplers.jl")

function run_sampler(my_model, lambda, max_attempts, mb_size, Print=false, prob_het=0.98, adapt_speed="none") 
    
    d, Nobs = size(my_model.ll.X)
    
    # Define minibatch sampler list:
    gs = Array{mbsampler}(d)
    gs[1] = umbsampler(0, Nobs, mb_size)

    for i in 2:d
        weights_het = abs.(my_model.ll.X[i,:])./sum(abs.(my_model.ll.X[i,:]))
        if length(my_model.ll.X[i,:].nzind) < length(my_model.ll.X[i,:]) 
            gs[i] = spwumbsampler(Nobs, mb_size, weights_het, prob_het)
        else 
            gs[i] = wumbsampler(Nobs, mb_size, weights_het)
        end
    end
    gs_list = mbsampler_list(d,gs);
    
    # Define output scheduler etc:
    A_xi = eye(d)
    #A_hyp = eye(hyperparam_size(my_model.pr))
    A_hyp = ones(1,hyperparam_size(my_model.pr))/hyperparam_size(my_model.pr)

    opf = projopf(A_xi, A_hyp, 10^3)
    opt = maxa_opt(max_attempts)
    outp = outputscheduler(opf,opt)
    bb = linear_bound(my_model.ll, my_model.pr, gs_list)
    mstate = zz_state(d)
    update_bound(bb, my_model.ll, my_model.pr, gs_list, mstate)
    
    # Define block Gibbs sampler:
    L = 1
    my_zz_sampler = zz_sampler(0, gs_list, bb, L, adapt_speed)
    hyper_sampler = block_gibbs_sampler(lambda)
    blocksampler = Array{msampler}(2)
    blocksampler[1] = my_zz_sampler
    blocksampler[2] = hyper_sampler
    
    # Run sampler:
    ZZ_block_sample(my_model, outp, blocksampler, mstate, Print)
    
    # Save data in files:
    filename  = "/xtmp/GZZ_data/shrinkage_prior/synthetic_data/lambda:"*string(lambda)*"-d:"*string(d)*"-Nobs:"*string(Nobs)*"-mb_size:"*string(mb_size)*".jld"
    save(filename, "xt_skeleton", outp.opf.xi_skeleton, "bt_skeleton", outp.opf.bt_skeleton, "hyper_skeleton", outp.opf.hyper_skeleton, "alpha_skeleton", outp.opf.alpha_skeleton)
    
    outp.opf.xi_skeleton, outp.opf.bt_skeleton, outp.opf.hyper_skeleton, outp.opf.alpha_skeleton = [], [], [], []
    gc()
end










