using JLD
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/zz_samplers.jl")

function run_sampler(my_model, lambda, max_attempts, mb_size, Print=false, prob_het=0.98, adapt_speed="by_var") 
    
    dim_cov, n_groups = my_model.pr.d, my_model.pr.K
    dim_total, Nobs = size(my_model.ll.X)
    group_size = Int(Nobs/n_groups)
    
    # Define minibatch sampler list:
    gs = Array{mbsampler}(dim_total)
    gs[1] = umbsampler(0, Nobs, mb_size)
    for i in 2:dim_total
        weights_het = abs.(X[i,:])./sum(abs.(X[i,:]))
        if length(X[i,:].nzind) < length(X[i,:])
            gs[i] = spwumbsampler(Nobs, mb_size, weights_het, prob_het)
        else 
            gs[i] = wumbsampler(Nobs, mb_size, weights_het)
        end
    end
    gs[n_groups+2] = umbsampler(0, Nobs, mb_size)
    gs_list = mbsampler_list(dim_total,gs)
    
    # Define output scheduler etc:
    A_xi = eye(dim_total)
    A_hyp = eye(hyperparam_size(my_model.pr))

    opf = projopf(A_xi, A_hyp, 10^3)
    opt = maxa_opt(max_attempts)
    outp = outputscheduler(opf,opt)
    bb = linear_bound(my_model.ll, my_model.pr, gs_list)
    mstate = zz_state(dim_total)
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
    filename  = "/xtmp/GZZ_data/mixed_effects/lambda:"*string(lambda)*"-dim_cov:"*string(dim_cov)*"-n_groups:"*string(n_groups)*"-Nobs:"*string(Nobs)*"-mb_size:"*string(mb_size)*".jld"
    save(filename, "xt_skeleton", outp.opf.xi_skeleton, "bt_skeleton", outp.opf.bt_skeleton, "hyper_skeleton", outp.opf.hyper_skeleton, "alpha_skeleton", outp.opf.alpha_skeleton)
    
    outp.opf.xi_skeleton, outp.opf.bt_skeleton, outp.opf.hyper_skeleton, outp.opf.alpha_skeleton = [], [], [], []
    gc()
    
end










