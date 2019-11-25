using Distributions, Optim, RCall, ProgressMeter

include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/mbsampler.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/types.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/structs.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/models.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/priors.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/bounds.jl")


function set_zz_samples(mstate::zz_state) 
    samples = zeros(length(mstate.ξ), 500)
    samples[:,1] = mstate.ξ
    return samples
end

zz_samples(mstate, h) = zz_samples(set_zz_samples(mstate), 1, 0., h)

function extend(msamples::gsamples, newsample) 
    msamples.tcounter += 1
    if msamples.tcounter > size(msamples.samples,2) 
        msamples.samples = extend_skeleton_points(msamples.samples, 500)
    end
    msamples.samples[:,msamples.tcounter] = newsample
    msamples.lastsamplet += msamples.h
end

function feed(msamples::zz_samples, mstate::zz_state, time::Float64, τ::Float64) #time is the current time of the sampler
    if (time + τ) > (msamples.lastsamplet + msamples.h)
        # extract sample 
        sampled_ξ = mstate.ξ + mstate.θ.*mstate.α*(msamples.lastsamplet+msamples.h-time)
        extend(msamples, sampled_ξ) 
    end
end

function set_hyp_samples(prior::gaussian_prior) 
    samples = zeros(hyperparam_size(prior), 500)
    samples[:,1] = get_hyperparameters(prior)
    return samples
end

hyp_samples(prior, h) = hyp_samples(set_hyp_samples(prior), 1, 0., h)


function feed(msamples::hyp_samples, prior::gaussian_prior, time::Float64, τ::Float64)
    if (time + τ) > (msamples.lastsamplet + msamples.h)
        # extract sample 
        sampled_hyp = get_hyperparameters(prior)
        extend(msamples, sampled_hyp) 
    end
end

function finalize(msamples::gsamples) 
    msamples.samples = msamples.samples[:,1:msamples.tcounter]
end
        

function feed(outp::outputscheduler, mstate::zz_state, prior::prior_model, time::Float64, bounce::Bool)
    
    if add_output(outp.opf, mstate, time, bounce)
        outp.opf.tcounter +=1 
        outp.opf.tot_bounces += 1
        if outp.opf.tcounter > size(outp.opf.bt_skeleton,2)
            outp.opf.xi_skeleton = extend_skeleton_points(outp.opf.xi_skeleton, outp.opf.size_increment)
            outp.opf.bt_skeleton = extend_skeleton_points(outp.opf.bt_skeleton, outp.opf.size_increment)
            outp.opf.hyper_skeleton = extend_skeleton_points(outp.opf.hyper_skeleton, outp.opf.size_increment)
            outp.opf.alpha_skeleton = extend_skeleton_points(outp.opf.alpha_skeleton, outp.opf.size_increment)
            outp.opf.vel_skeleton = extend_skeleton_points(outp.opf.vel_skeleton, outp.opf.size_increment)
        end
        outp.opf.xi_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, mstate.ξ)
        outp.opf.bt_skeleton[:,outp.opf.tcounter] = time
        outp.opf.hyper_skeleton[:,outp.opf.tcounter] = compress_hyp(outp.opf, get_hyperparameters(prior))
        outp.opf.alpha_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, mstate.α)
        outp.opf.vel_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, mstate.α.*mstate.θ)
        
        outp.opf.theta = mstate.θ
        outp.opf.n_bounces = mstate.n_bounces
        
        update_summary(outp, mstate)
    end
    
    if to_trim(outp.opt) 
        trim(outp.opf, mstate)
        time = 0
    end
    outp.opt = eval_stopping(outp.opt, mstate.ξ, time, bounce)
    return outp, time
end


function is_running(opt::outputtimer)
    return opt.running
end


function add_output(opf::outputformater, state::zz_state, time::Float64, bounce::Bool)
   return bounce 
end

function compress_xi(opf::outputformater, xi)
   return xi 
end

function extend_skeleton_points(skeleton_points, extension=1000)
    m, n = size(skeleton_points)
    skeleton_new = zeros(m, n+extension)
    skeleton_new[:,1:n] = skeleton_points
    return skeleton_new
end 

#--------------------------------------------------------------------------------------------------------

function finalize(opf::outputformater)
    opf.xi_skeleton = opf.xi_skeleton[:,1:opf.tcounter]
    opf.bt_skeleton = opf.bt_skeleton[:,1:opf.tcounter]
    opf.hyper_skeleton = opf.hyper_skeleton[:,1:opf.tcounter]
    opf.alpha_skeleton = opf.alpha_skeleton[:,1:opf.tcounter]
    opf.vel_skeleton = opf.vel_skeleton[:,1:opf.tcounter]
end



function trim(opf::projopf, mstate::zz_state)
    
    xi_skeleton = zeros(size(opf.xi_skeleton,1), 1000)
    bt_skeleton = zeros(size(opf.bt_skeleton,1), 1000)
    alpha_skeleton = zeros(size(opf.alpha_skeleton,1), 1000)
    hyper_skeleton = zeros(size(opf.hyper_skeleton,1), 1000)
    vel_skeleton = zeros(size(opf.vel_skeleton,1), 1000)

    xi_skeleton[:,1] = copy(opf.xi_skeleton[:,opf.tcounter])
    alpha_skeleton[:,1] = copy(opf.alpha_skeleton[:,opf.tcounter])
    hyper_skeleton[:,1] = copy(opf.hyper_skeleton[:,opf.tcounter])
    vel_skeleton[:,1] = copy(opf.vel_skeleton[:,opf.tcounter])
    
    opf.xi_skeleton = copy(xi_skeleton)
    opf.bt_skeleton = copy(bt_skeleton)
    opf.alpha_skeleton = copy(alpha_skeleton)
    opf.hyper_skeleton = copy(hyper_skeleton)
    opf.vel_skeleton = copy(vel_skeleton)
    
    opf.xi_lastbounce = copy(mstate.ξ)
    opf.tcounter = 1
end

function update_summary(outp::outputscheduler, mstate::zz_state)
    
    if outp.opt.acounter > outp.opt.discard
        vel = mstate.θ.*mstate.α
        current_t = outp.opf.bt_skeleton[1,outp.opf.tcounter]
        ΔT = current_t - outp.opf.T_lastbounce
        outp.opf.xi_mu = (outp.opf.T_lastbounce*outp.opf.xi_mu + outp.opf.xi_lastbounce*ΔT + 1/2*vel*ΔT^2)/current_t
        outp.opf.xi_m2 = (outp.opf.T_lastbounce*outp.opf.xi_m2 + outp.opf.xi_lastbounce.^2*ΔT + vel.*outp.opf.xi_lastbounce*ΔT^2 + 1/3*vel.^2*ΔT^2)/current_t
        
        outp.opf.T_lastbounce = copy(current_t)
        outp.opf.xi_lastbounce = copy(mstate.ξ)
    end 
end




function built_projopf(A_xi, A_hyp, size_increment)
    d_out_xi, d = size(A_xi)
    d_out_hyp = size(A_hyp,1)
    xi_skeleton = zeros(d_out_xi, 10*size_increment)
    bt_skeleton = zeros(1, 10*size_increment)
    tcounter = 1
    theta = ones(d)
    hyper_skeleton = ones(d_out_hyp, 10*size_increment)
    alpha_skeleton = ones(d_out_xi, 10*size_increment)
    vel_skeleton = ones(d_out_xi, 10*size_increment)
    n_bounces = zeros(d)
    xi_mu = zeros(d)
    xi_m2 = zeros(d)
    xi_lastbounce = zeros(d)
    T_lastbounce = 0.
    tot_bounces = 1
    return d, xi_skeleton, bt_skeleton, theta, alpha_skeleton, vel_skeleton, n_bounces, hyper_skeleton, tcounter, size_increment, A_xi, d_out_xi, A_hyp, d_out_hyp, xi_mu, xi_m2, xi_lastbounce, T_lastbounce, tot_bounces
end

function compress_xi(outp::projopf, xi)
   return outp.A_xi * xi  
end

function compress_hyp(outp::projopf, hyperparameter)
   return outp.A_hyp * hyperparameter  
end


function to_trim(opt::maxa_opt)
    if opt.acounter == opt.discard 
        print("Trimmed after ", opt.acounter, " bouncing attempts \n")
        return true 
    else
        return false
    end
end


function eval_stopping(opt::outputtimer, xi, time, bounce)
    opt.acounter+=1
    if opt.acounter >= opt.max_attempts
        opt.running = false
    end
    return opt
end

#--------------------------------------------------------------------------------------------
# ----------------------------- UPDATE STEPS FOR HYPERPARAMETERS ----------------------------
#--------------------------------------------------------------------------------------------

function get_event_time(mysampler::block_gibbs_sampler, mstate::zz_state, model::model)
    return rand(Exponential(1.0/mysampler.λ))
end

function evolve_path(mysampler::block_gibbs_sampler, mstate::zz_state, τ)
    mstate.ξ += τ*mstate.θ.*mstate.α
    mstate.T += τ
end

function update_state(mysampler::block_gibbs_sampler, mstate::zz_state, model::model, τ)
    block_Gibbs_update_hyperparams(model.pr, mstate.ξ)
    mysampler.nbounces += 1
    return true
end

#--------------------------------------------------------------------------------------------
# ------------------------------- UPDATE STEPS FOR PARAMETERS -------------------------------
#--------------------------------------------------------------------------------------------


function get_event_time(mysampler::zz_sampler, mstate::zz_state, model::model)
    d = length(mstate.ξ)
    update_bound(mysampler.bb, model.ll, model.pr, mysampler.gs, mstate)
    event_times = [get_event_time(mysampler.bb.a[i], mysampler.bb.b[i]) for i in 1:d]  
    τ, i0 = findmin(event_times) 
    mysampler.i0 = i0
    return τ
end

function evolve_path(mysampler::zz_sampler, mstate::zz_state, τ)
    mstate.ξ += τ*mstate.θ.*mstate.α
    mstate.T += τ
end

function update_state(mysampler::zz_sampler, mstate::zz_state, model::model, τ)
    mb = gsample(mysampler.gs.mbs[mysampler.i0])
    rate_estimated = estimate_rate(model, mstate, mysampler.i0, mb, mysampler.gs)
    bound = evaluate_bound(mysampler.bb, τ, mysampler.i0)
    
    alpha = rate_estimated/bound
    #print(alpha, "\n")
    if alpha > 1 + 1e-10
        print("alpha: ", alpha, "\n")
        error(rate_estimated, " | ", 
              bound, " | ", 
              τ, " | ", mstate.α[mysampler.i0], " | ", mysampler.i0, " | ",
              mysampler.bb.a[mysampler.i0], ", ", mysampler.bb.b[mysampler.i0])
    end
    bounce = false
    if rand() < alpha
        vel = mstate.θ.*mstate.α
        ΔT = mstate.T - mstate.T_lastbounce
        
        mstate.mu = (mstate.T_lastbounce*mstate.mu + mstate.ξ_lastbounce*ΔT + 1/2*vel*ΔT^2)/mstate.T
        mstate.m2 = (mstate.T_lastbounce*mstate.m2 + mstate.ξ_lastbounce.^2*ΔT + vel.*mstate.ξ_lastbounce*ΔT^2 + 1/3*vel.^2*ΔT^2)/mstate.T
        
        mstate.T_lastbounce = copy(mstate.T)
        mstate.ξ_lastbounce = copy(mstate.ξ)
        
        mstate.θ[mysampler.i0] *= -1
        bounce = true
        mstate.n_bounces[mysampler.i0] += 1
        
        
        #adapt speed: 
        if mysampler.adapt_speed == "by_bounce" 
            if (sum(mstate.n_bounces)%mysampler.L == 0)  & (sum(mstate.n_bounces) >= 1)
                segment_idx = Int64(sum(mstate.n_bounces)/mysampler.L) 
                est_segment = mstate.n_bounces ./ mstate.α
                est_segment /= sum(est_segment)
                mstate.est_rate = (segment_idx*mstate.est_rate + est_segment)/(segment_idx+1)
                mstate.α = 1./mstate.est_rate
                mstate.α[1] = max(minimum(mstate.α[2:end])/10^2, mstate.α[1]) 

                #if minimum(mstate.n_bounces) >= 5 
                #   mstate.α ./=  (mstate.n_bounces).^0.35
                #end
            end
        elseif mysampler.adapt_speed == "by_var"  
            if minimum(mstate.m2 - mstate.mu.^2) > 0 
                mstate.α = sqrt.(mstate.m2 - mstate.mu.^2) 
            end
        end
        mstate.α /= mean(mstate.α)
        
    end 
    return bounce
end

#--------------------------------------------------------------------------------------------
# -------------------------------------- MAIN SAMPLER ---------------------------------------
#--------------------------------------------------------------------------------------------

function ZZ_block_sample(model::model, outp::outputscheduler, blocksampler::Array{msampler}, mstate::zz_state, Print=true)

    K = length(blocksampler)
    counter = 1

    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter])
    
    # run sampler:
    start = time()
    bounce = false
    while(is_running(outp.opt))
                
        τ_list = [get_event_time(blocksampler[i], mstate, model) for i in 1:K]
        τ, k0 = findmin(τ_list)
        t += τ 
        
        evolve_path(blocksampler[k0], mstate, τ)
        bounce = update_state(blocksampler[k0], mstate, model, τ)
        outp, t = feed(outp, mstate, model.pr, t, bounce)
        
        counter += 1
        if counter%10_000 == 0 
            gc()
        end
        if counter % (outp.opt.max_attempts/10) == 0 && Print 
            @printf("%i percent attempts in %.2f min; zz bounces = %i, hyp bounces = %i, total time of process = %.3f \n", Int64(100*counter/(outp.opt.max_attempts)), (time()-start)/60, sum(mstate.n_bounces), blocksampler[2].nbounces, mstate.T)
        end
    end
    finalize(outp.opf)
    return outp
end


#--------------------------------------------------------------------------------------------
# ------------------------------------ DISCRETE SAMPLER -------------------------------------
#--------------------------------------------------------------------------------------------

function ZZ_block_sample_discrete(model::model, opt::outputtimer, blocksampler::Array{msampler}, mstate::zz_state, xi_samples::zz_samples, pr_samples::hyp_samples)

    K = length(blocksampler)
    counter = 1
    t = 0.
    @assert xi_samples.h == pr_samples.h "Discretizations for samples and hypersamples do not match"
    
    # run sampler:
    start = time()
    bounce = false
    while(is_running(opt))
                
        τ_list = [get_event_time(blocksampler[k], mstate, model) for k in 1:K]
        τ, k0 = findmin(τ_list)
        feed(xi_samples, mstate, t, τ)
        feed(pr_samples, model.pr, t, τ)
        
        t += τ 
        evolve_path(blocksampler[k0], mstate, τ)
        bounce = update_state(blocksampler[k0], mstate, model, τ)
        eval_stopping(opt, mstate.ξ, t, bounce)
        
        counter += 1
        if counter%10_000 == 0 
            gc()
        end
        if counter % (opt.max_attempts/10) == 0 
            print(Int64(100*counter/(opt.max_attempts)), "% attempts in ", round((time()-start)/60, 2), " min, zz bounces = ", sum(mstate.n_bounces), ", hyp bounces = ", blocksampler[2].nbounces, ", samples extracted = ", size(xi_samples.samples,2), "\n")
            
        end
        finalize(xi_samples)
        finalize(pr_samples)
    end
end



#--------------------------------------------------------------------------------------------
# --------------------------------------- OTHER STUFF ---------------------------------------
#--------------------------------------------------------------------------------------------

function extract_samples(skeleton_points, bouncing_times, h, interpolation="linear") 
    d, n = size(skeleton_points)
    path_length = bouncing_times[end] - bouncing_times[1]
    n_samples = Int64(floor(path_length/h)) + 1
    samples = zeros(d, n_samples)
    samples[:,1] = skeleton_points[:,1] 
    sample_index = 2
    time_location = bouncing_times[1] + h
    
    for i in 1:n-1
        start, stop = skeleton_points[:,i], skeleton_points[:,i+1] 
        Δ_pos = stop - start   
        Δ_T = bouncing_times[i+1] - bouncing_times[i]
        while time_location <= bouncing_times[i+1]
            if interpolation == "linear"
                samples[:,sample_index] = start + Δ_pos/Δ_T*(time_location - bouncing_times[i])
            elseif interpolation == "constant"
                samples[:,sample_index] = start
            end
            time_location += h
            sample_index += 1
        end
    end
    return samples
end

function compute_configT(m::model, samples::Array{Float64}, k)
    d, Nobs = size(X) 
    n_samples = size(samples,2)
    configT = 0.0
    for i in 1:n_samples
        configT += samples[k,i]*partial_derivative(m::model, samples[:,i], k)
    end
    return configT/n_samples
end


function find_root(my_model::model, ξ_0)
    d, Nobs = size(my_model.ll.X)
    function gradient!(F, ξ)
        F[:] = gradient(my_model, ξ) 
    end
    neg_log_posterior(ξ) = - log_posterior(my_model, ξ)  
    result = optimize(neg_log_posterior, gradient!, ξ_0, LBFGS())
    root = result.minimizer
    return root
end



function stochastic_gradient(m::model, ξ, batch_size) 
    d = length(ξ)
    # pick random minibatch 
    mb = Int.(floor.(my_model.ll.Nobs*rand(batch_size)))+1
    return [(m.ll.Nobs*mean(partial_derivative_vec(m.ll, ξ_0, k, mb)) 
             + partial_derivative(m.pr, ξ_0, k)) for k in 1:d]
end

function SGD(m::model, ξ_0, batch_size, γ, tol) 
    d = length(ξ_0) 
    ξ_current = zeros(d)
    ξ_updated = copy(ξ_0)
    @showprogress for iter in 1:10^4  
        ξ_updated = ξ_current - γ*stochastic_gradient(m, ξ_current, batch_size)
        if norm(ξ_updated-ξ_current) < tol 
            @printf("converged in %f iterations", iter)
            break;
        else 
            ξ_current = copy(ξ_updated)
        end
    end
    return ξ_current
end


function acf(x, maxlag)
    n = size(x)[1]
    acf_vec = zeros(maxlag)
    xmean = mean(x)
    for lag in 1:maxlag
        index, index_shifted = 1:(n-lag), (lag+1):n
        acf_vec[lag] = mean((x[index]-xmean).*(x[index_shifted]-xmean))
    end
    acf_vec/var(x)
end


getBytes(x::DataType) = sizeof(x);

function getBytes(x)
   total = 0;
   fieldNames = fieldnames(typeof(x));
   if fieldNames == []
      return sizeof(x);
   else
     for fieldName in fieldNames
        total += getBytes(getfield(x,fieldName));
     end
     return total;
   end
end


#--------------------------------------------------------------------------------------------
# --------------------------------------- [OLD STUFF] ---------------------------------------
#--------------------------------------------------------------------------------------------


function GZZ_sample(m::model, 
                    outp::outputscheduler, 
                    gs_list::sampler_list, 
                    out_samples::gzz_samples,
                    T_gibbs::Int64, 
                    n_gibbs::Int64,
                    update_hyper=true)  #last argument is for sanity checks
    
    d, Nobs = size(m.ll.X) 
    ξ_samples = zeros(d,T_gibbs)
    hyper_samples = Array{gaussian_prior}(T_gibbs)
    hyper_samples[1] = m.pr
    
    @showprogress for t in 1:T_gibbs
        outp.opt = maxa_opt(n_gibbs)
        ZZ_sample(m, outp, gs_list)
        ξ = outp.opf.xi_skeleton[:,outp.opf.tcounter]
        if update_hyper
            block_Gibbs_update_hyperparams(m.pr, ξ)  # this updates my_model; to check, run:
    #     print(my_model.pr, "\n")
        end
        out_samples.xi_samples[:,t] = outp.opf.xi_skeleton[:,outp.opf.tcounter]
        out_samples.hyper_samples[t] = deepcopy(m.pr)
        out_samples.zz_nbounces[t] = outp.opf.tcounter
    end
    finalize(outp.opf)
end

function compute_configT(m::model, samples::gzz_samples, k)
    d, Nobs = size(m.ll.X)
    n_samples = length(samples.hyper_samples)
    configT = 0.0
    for i in 1:n_samples
        m.pr = samples.hyper_samples[i]
        configT += samples.xi_samples[k,i]*partial_derivative(m::model, samples.xi_samples[:,i], k)
    end
    return configT/n_samples
end

function compute_configT(m::model, xi_samples::Array{Float64}, hyper_samples::Array{Float64}, k)
    d, Nobs = size(m.ll.X)
    n_samples = size(xi_samples,2)
    configT = 0.0
    for i in 1:n_samples
        set_hyperparams(m.pr, hyper_samples[:,i])     
        configT += xi_samples[k,i]*partial_derivative(m::model, xi_samples[:,i], k)
    end
    return configT/n_samples
end

function compute_ESS(opf::outputformater, B::Int64) 
    dim = size(opf.xi_skeleton,1)
    T = opf.bt_skeleton[1,opf.tcounter]
    
    batch_length = T/B
    Y = zeros(B, dim)
    t = opf.bt_skeleton[1,1]
    xi = opf.xi_skeleton[:,1]
    vel = opf.vel_skeleton[:,1]
    
    k = 1 #counter for skeleton point
    
    @showprogress for i in 1:B
        while t < i*T/B 
            next_bounce_time = min(opf.bt_skeleton[1,k+1], i*T/B)
            Δt = next_bounce_time - t
            Y[i,:] += xi*Δt + vel.*Δt.^2/2
            t += Δt 
            if next_bounce_time == opf.bt_skeleton[1,k+1] 
                xi = opf.xi_skeleton[1,k+1] 
                vel = opf.vel_skeleton[1,k+1]
                k += 1
            else 
                xi += vel.*Δt
            end
        end
    end
    Y *= sqrt(B/T)
    
    var1 = opf.xi_m2 - opf.xi_mu.^2
    var2 = zeros(dim)
    for i in 1:dim 
        var2[i] = var(Y[:,dim])
    end
    ESS = T*var1./var2
end


#--------------------------------------------------------------------------------------------
# ---------------------------------------- GIBBS HMC ----------------------------------------
#--------------------------------------------------------------------------------------------

function HMC(model::model, current_q, epsilon, L, Metropolise=true) 
    
    q = copy(current_q)
    p = randn(length(current_q))
    current_p = copy(p)
    
    p -= epsilon*gradient(model,q)/2
    for i in 1:L 
        q += epsilon*p 
        if i!=L 
            p -= epsilon*gradient(model,q)
        end
    end 
    p -= epsilon*gradient(model,q)/2
    
    p = -p 
    
    current_U = -log_posterior(model, current_q) 
    current_K = sum(current_p.^2)/2
    proposed_U = -log_posterior(model, q) 
    proposed_K = sum(p.^2)/2
    
    if Metropolise 
        if rand() < exp(current_U-proposed_U+current_K-proposed_K) 
            return q, 1
        else 
            return current_q, 0 
        end
    else
        return q, 1
    end
end

function GibbsHMC(model::model, ξ0, epsilon, L, T, Metropolise=true, Print=true) 
    d = size(model.ll.X,1)
    d_hyp = hyperparam_size(model.pr)
    xi_samples = zeros(d,T+1)
    xi_samples[:,1] = ξ0
    hyper_samples = zeros(d_hyp,T+1)
    hyper_samples[:,1] = get_hyperparameters(model.pr)
    
    HMC_accept = 0
    
    start = time()
    for t in 1:T 
        hmc = HMC(model, xi_samples[:,t], epsilon, L, Metropolise)
        xi_samples[:,t+1] = hmc[1]
        HMC_accept += hmc[2]
        block_Gibbs_update_hyperparams(model.pr, xi_samples[:,t+1])
        hyper_samples[:,t+1] = get_hyperparameters(model.pr)
        
        if Print 
            if t % (T/10) == 0 
                @printf("%i percent steps in %.1f min; HMC acceptance = %i percent \n", Int64(100*t/T), (time()-start)/60, 100*HMC_accept/t)
            end
        end
    end
    if !Print 
        print("HMC acceptance = ", 100*HMC_accept/T, " percent; ")
    end
    return xi_samples, hyper_samples, HMC_accept/T
end 








