using Distributions, Optim, RCall, ProgressMeter
include("mbsampler.jl")
R"""
library("GIGrvg")
"""

abstract type outputtimer end
abstract type outputformater end

abstract type ll_model end
abstract type prior_model end
abstract type msampler end
    
# Define abstract type for gaussian prior, sub-types of this abstract types must have attributes mu and sigma2  
abstract type gaussian_prior <:prior_model end
abstract type laplace_prior <:prior_model end

abstract type bound end


mutable struct const_bound<:bound
    a::Float64
end

mutable struct linear_bound<:bound
    a_fixed::Array{Float64}
    b_fixed::Array{Float64}
    a_xi::Array{Float64}
    b_xi::Array{Float64}
end

mutable struct outputscheduler
    opf::outputformater
    opt::outputtimer
end

mutable struct zz_state 
    ξ::Array{Float64}
    θ::Array{Float64}
end

mutable struct zz_sampler <:msampler
   i0::Int64
   gs::sampler_list
   bb::bound
end

function feed(outp::outputscheduler, state::zz_state, prior::prior_model, time::Float64, bounce::Bool)
    
    if add_output(outp.opf, state, time, bounce)
        if outp.opf.tcounter > size(outp.opf.bt_skeleton,2)
            outp.opf.xi_skeleton = extend_skeleton_points(outp.opf.xi_skeleton, outp.opf.size_increment)
            outp.opf.bt_skeleton = extend_skeleton_points(outp.opf.bt_skeleton, outp.opf.size_increment)
            outp.opf.hyper_skeleton = extend_skeleton_points(outp.opf.hyper_skeleton, outp.opf.size_increment)
        end
        outp.opf.xi_skeleton[:,opf.tcounter] = compress_xi(outp.opf, state.ξ)
        outp.opf.bt_skeleton[:,opf.tcounter] = time
        outp.opf.hyper_skeleton[:,opf.tcounter] = get_hyperparams(prior)
        outp.opf.tcounter +=1 
    end
    outp.opt = eval_stopping(outp.opt, state.ξ, time, bounce)
    return outp
end

#--------------------------------------------------------------------------------------------------------


function is_running(opt::outputtimer)
    return opt.running
end

#--------------------------------------------------------------------------------------------------------



function add_output(opf::outputformater, state::zz_state, time::Float64, bounce::Bool)
   return bounce 
end

function compress_xi(opf::outputformater, xi)
   return xi 
end

function extend_skeleton_points(skeleton_points, extension=100)
    m, n = size(skeleton_points)
    skeleton_new = zeros(m, n+extension)
    skeleton_new[:,1:n] = skeleton_points
    return skeleton_new
end 

#--------------------------------------------------------------------------------------------------------

function finalize(opf::outputformater)
    opf.xi_skeleton = opf.xi_skeleton[:,1:opf.tcounter-1]
    opf.bt_skeleton = opf.bt_skeleton[:,1:opf.tcounter-1]
    opf.hyper_skeleton = opf.hyper_skeleton[:,1:opf.tcounter-1]
end

mutable struct projopf <:outputformater
    d::Int64
    xi_skeleton::Array{Float64}
    bt_skeleton::Array{Float64}
    theta::Array{Float64} 
    hyper_skeleton::Array{Float64}
    hyper_size::Int64
    tcounter::Int64
    size_increment::Int64
    A::Array{Float64}
    d_out::Int64
end

projopf(A, size_increment, hyper_size) = projopf(built_projopf(A, size_increment, hyper_size)...)
projopf(A, size_increment) = projopf(built_projopf(A, size_increment, 0)...)

function built_projopf(A, size_increment, hyper_size)
    d_out, d = size(A)
    xi_skeleton = zeros(d,10*size_increment)
    bt_skeleton = zeros(1,10*size_increment)
    tcounter = 2
    theta = ones(d)
    hyper_skeleton = ones(hyper_size, 10*size_increment)
    return d, xi_skeleton, bt_skeleton, theta, hyper_skeleton, hyper_size, tcounter, size_increment, A, d_out
end

function compress_xi(outp::projopf, xi)
   return outp.A * xi  
end

#--------------------------------------------------------------------------------------------------------

mutable struct maxa_opt <:outputtimer
    running::Bool
    max_attempts::Int64
    acounter::Int64
end
maxa_opt(max_attempts) = maxa_opt(true, max_attempts, 1)

function eval_stopping(opf::maxa_opt, xi, time, bounce)
    opf.acounter+=1
    if opf.acounter >= opf.max_attempts
        opf.running = false
    end
    return opf
end




#--------------------------------------------------------------------------------------------------------
# Models
#--------------------------------------------------------------------------------------------------------
mutable struct model
    ll::ll_model
    pr::prior_model
end





# #--------------------------------------------------------------------------------------------------------
# Derivative for model = prior + likelihood 
# #--------------------------------------------------------------------------------------------------------

function log_posterior(m::model, ξ) 
    return log_likelihood(m.ll, ξ) + log_prior(m.pr, ξ) 
end

function gradient(m::model, ξ) 
    return gradient(m.ll, ξ) + gradient(m.pr, ξ) 
end

function partial_derivative_vec(m::model, ξ, k, mb) 
    return partial_derivative_vec(m.ll, ξ, k, mb) + partial_derivative(m.pr, ξ, k)/m.ll.Nobs
end

function partial_derivative(m::model, ξ, k) 
    Nobs = length(m.ll.y)
    return sum(partial_derivative_vec(m, ξ, k, 1:Nobs))
end




# #--------------------------------------------------------------------------------------------------------
# Derivative for likelihood 
# #--------------------------------------------------------------------------------------------------------

function log_likelihood(ll::ll_model, ξ)
    d = length(ξ)
    return sum(log_likelihood_vec(ll, ξ, 1:ll.Nobs))
end



function gradient(ll::ll_model, ξ) 
    d = length(ξ)
    return [sum(partial_derivative_vec(ll::ll_model, ξ, k, 1:ll.Nobs)) for k in 1:d]
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::mbsampler)
    return ll.Nobs*sum(partial_derivative_vec(ll, ξ, k, mb).*get_ubf(gs,mb))
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::sampler_list)
    return estimate_ll_partial(ll, ξ, k, mb, gs.mbs[k])
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::cvmbsampler_list)
    return gs.gradient_root_sum[k] +  ll.Nobs*sum((partial_derivative_vec(ll, ξ, k, mb) 
                                             - gs.gradient_root_vec[k,mb]).*get_ubf(gs.mbs[k],mb))
end

function estimate_rate(m::model, ξ, θ, i0, mb, gs::sampler_list)
    rate_prior = pos(θ[i0]*partial_derivative(m.pr, ξ, i0))
    rate_likelihood = pos(θ[i0]*estimate_ll_partial(m.ll, ξ, i0, mb, gs.mbs[i0]))
    return rate_prior + rate_likelihood
end

function estimate_rate(m::model, ξ, θ, i0, mb, gs::cvmbsampler_list)
    rate_1 = pos(θ[i0]* ( gs.gradient_log_posterior_root_sum[i0] + partial_derivative(m.pr, ξ, i0) - gs.gradient_log_prior_root[i0] ))
    rate_2 = pos( θ[i0]*m.ll.Nobs*sum((partial_derivative_vec(m.ll, ξ, i0, mb) 
                                      - gs.gradient_log_ll_root_vec[i0,mb]).*get_ubf(gs.mbs[i0],mb)) )
    return rate_1 + rate_2
end

#--------------------------------------------------------------------------------------------------------
# Derivative for likelihood of logistic regression
#--------------------------------------------------------------------------------------------------------

struct ll_logistic<:ll_model
    X::Array{Float64}
    y::Array{Int64}
    Nobs::Int64
end
ll_logistic(X, y) = ll_logistic(X, y, length(y)) 

function log_likelihood_vec(ll::ll_logistic, ξ, mb)
   return - ( log.(1 + vec(exp.(ξ'll.X[:,mb]))) - ll.y[mb] .* vec(ξ'll.X[:,mb]) )
end
function partial_derivative_vec(ll::ll_logistic, ξ, k, mb)
    expX = exp.(ξ'll.X[:,mb]);
    return ll.X[k,mb].* ( vec(expX./(1+expX)) - ll.y[mb] )
    expX = nothing 
    gc();
end

#------------------------- Bounds, no control variates -----------------------------# 

function build_linear_bound(ll::ll_logistic, pr::gaussian_prior, gs::mbsampler_list)
    d, Nobs = size(ll.X)
    σ2 = get_σ2(pr)
    a_fixed = [maximum(abs.(ll.X[i,:]./get_weights(gs.mbs[i],1:Nobs))) for i in 1:d]
    b_fixed = ones(d)./σ2
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_logistic, pr::gaussian_prior, gs::mbsampler_list, ξ0, θ)
    bb.a_xi = abs.(ξ0-get_μ(pr))./get_σ2(pr)
    bb.b_xi = zeros(d)
    return bb
end

#------------------------- Bounds, with control variates -----------------------------# 

function build_linear_bound(ll::ll_logistic, pr::gaussian_prior, gs::cvmbsampler_list)
    
    d, Nobs = size(ll.X)
    C_lipschitz = zeros(d, Nobs)
    for j in 1:Nobs
        C_lipschitz[:,j] = 1/4*[abs.(ll.X[i,j])*norm(ll.X[:,j]) for i in 1:d]
    end
    σ2 = get_σ2(pr)
    C = [maximum(C_lipschitz[i,:]./get_weights(gs.mbs[i], 1:Nobs)) for i in 1:d] + 1.0 ./σ2
    a_fixed = zeros(d)
    b_fixed = √d*C
    
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_logistic, pr::gaussian_prior, gs::cvmbsampler_list, ξ0, θ)
    d = length(ξ0)
    norm_ = norm(gs.root-ξ0)
    bb.a_xi = [pos(θ[i]*gs.gradient_log_posterior_root_sum[i]) + norm_*bb.b_fixed[i]/√d for i in 1:d]
    bb.b_xi = zeros(d)
    return bb
end

#--------------------------------------------------------------------------------------------------------
# Derivative for likelihood of logistics regression
#--------------------------------------------------------------------------------------------------------

struct ll_logistic_sp<:ll_model
    X::SparseMatrixCSC
    y::Array{Int64}
    Nobs::Int64
end
ll_logistic_sp(X, y) = ll_logistic_sp(X, y, length(y)) 

function log_likelihood_vec(ll::ll_logistic_sp, ξ, mb)
   return - ( log.(1 + vec(exp.(ξ'll.X[:,mb]))) - ll.y[mb] .* vec(ξ'll.X[:,mb]) )
end
#function partial_derivative_vec(ll::ll_logistic_sp, ξ, k, mb) 
    #mb_size = length(mb)
    #nz_ind = ll.X[k,mb].nzind
    #pd_vec = spzeros(mb_size)
    #pd_vec[nz_ind] = ll.X[k,nz_ind].* ( vec(exp.(ξ'll.X[:,nz_ind]) ./ (1+exp.(ξ'll.X[:,nz_ind]))) - ll.y[nz_ind] )
    #return ll.X[k,mb].* ( vec(exp.(ξ'll.X[:,mb]) ./ (1+exp.(ξ'll.X[:,mb]))) - ll.y[mb] )
    #return pd_vec
#end

function partial_derivative_vec(ll::ll_logistic_sp, ξ, k, mb) 
    mb_size = length(mb)
    nz_ind = ll.X[k,mb].nzind
    pd_vec = spzeros(mb_size)
    mb_nz_ind = mb[nz_ind]
    pd_vec[nz_ind] = ll.X[k,mb_nz_ind].* ( vec(exp.(ξ'll.X[:,mb_nz_ind]) ./ 
                                         (1+exp.(ξ'll.X[:,mb_nz_ind]))) - ll.y[mb_nz_ind] )
    return pd_vec
end

#-------------------- Bounds, no control variates + SPARSE -----------------------------# 

function build_linear_bound(ll::ll_logistic_sp, pr::gaussian_prior, gs::mbsampler_list)
    # Redefine 
    d, Nobs = size(ll.X)
    σ2 = get_σ2(pr)
    a_fixed = zeros(d)
    for i in 1:d
        nz_ind = ll.X[i,:].nzind
        a_fixed[i] = maximum(abs.(ll.X[i,nz_ind]./get_weights(gs.mbs[i],nz_ind) ))
    end
    b_fixed = ones(d)./σ2
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_logistic_sp, pr::gaussian_prior, gs::mbsampler_list, ξ0, θ)
    bb.a_xi = abs.(ξ0-get_μ(pr))./get_σ2(pr)
    bb.b_xi = zeros(d)
    return bb
end

#-------------------- Bounds, with control variates + SPARSE -----------------------------# 

function build_linear_bound(ll::ll_logistic_sp, pr::gaussian_prior, gs::cvmbsampler_list)
    
    d, Nobs = size(ll.X)
    C_lipschitz = spzeros(d, Nobs)
    C = zeros(d)
    normXj = [norm(ll.X[:,j]) for j in 1:Nobs]
    for i in 1:d 
        nz_ind = ll.X[i,:].nzind
        C_lipschitz[i,nz_ind] = 1/4*abs.(ll.X[i,nz_ind ]).*normXj[nz_ind]
        C[i] = maximum( C_lipschitz[i,nz_ind]./get_weights(gs.mbs[i], nz_ind) )
    end
    C += 1./get_σ2(pr)
    a_fixed = zeros(d)
    b_fixed = √d*C
    
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_logistic_sp, pr::gaussian_prior, gs::cvmbsampler_list, ξ0, θ)
    d = length(ξ0)
    norm_ = norm(gs.root-ξ0)
    bb.a_xi = [pos(θ[i]*gs.gradient_log_posterior_root_sum[i]) + norm_*bb.b_fixed[i]/√d for i in 1:d]
    bb.b_xi = zeros(d)
    return bb
end


#--------------------------------------------------------------------------------------------------------
# Derivative for 0-likelihood 
#--------------------------------------------------------------------------------------------------------

struct ll_zeros<:ll_model
    X::Array{Float64}
    y::Array{Int64}
    Nobs::Int64
end
ll_zeros(X, y) = ll_zeros(X, y, length(y)) 

function log_likelihood_vec(ll::ll_zeros, ξ, mb)
   return zeros(length(mb))
end

function partial_derivative_vec(ll::ll_zeros, ξ, k, mb) 
    return zeros(length(mb))
end

function build_linear_bound(ll::ll_zeros, pr::gaussian_prior, gs::mbsampler_list)
    d, Nobs = size(ll.X)
    σ2 = get_σ2(pr)
    a_fixed = zeros(d)
    b_fixed = ones(d)./σ2
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_zeros, pr::gaussian_prior, gs::mbsampler_list, ξ0, θ)
    d = length(ξ0)
    bb.a_xi = abs.(ξ0-get_μ(pr))./get_σ2(pr)
    bb.b_xi = zeros(d)
    return bb
end


#--------------------------------------------------------------------------------------------------------
# Derivative for prior 
#--------------------------------------------------------------------------------------------------------
        
function gradient(pp::prior_model, ξ) 
    d, = length(ξ)
    return [partial_derivative(pp, ξ, k) for k in 1:d]
end


function log_prior(pp::gaussian_prior, ξ) 
    return -0.5*sum( (ξ - get_μ(pp)).^2 ./ get_σ2(pp) )
end




function partial_derivative(pp::gaussian_prior, ξ, k) 
    return (ξ[k] - get_μ(pp)[k]) ./ (get_σ2(pp)[k])
end

#--------------------------------------------------------------------------------------------------------
# Structure implementing Gaussian non-hierarchical prior
#--------------------------------------------------------------------------------------------------------

struct gaussian_prior_nh <:gaussian_prior
    μ::Array{Float64}
    σ2::Array{Float64}
end

function get_σ2(pp::gaussian_prior)
    return pp.σ2
end

function get_μ(pp::gaussian_prior)
    return pp.μ
end

function hyperparams_size(pp::gaussian_prior) 
    return 0 
end

function block_Gibbs_update_hyperparams(g_prior_nh::gaussian_prior_nh, ξ) 
    # do nothing
end

#--------------------------------------------------------------------------------------------------------
# Structure implementing horseshoe prior
#--------------------------------------------------------------------------------------------------------

mutable struct HS_prior <:gaussian_prior
    d ::Int64
    σ02 ::Float64 # variance for the intercept 
    # hyperparameters:
    λ2 ::Array{Float64}
    τ2 ::Float64
    ν ::Array{Float64}
    γ ::Float64
end

HS_prior(d, σ02) = HS_prior(d, σ02, ones(d-1), 1., ones(d-1), 1.)

function get_σ2(prior::HS_prior)
    return vcat(prior.σ02, prior.λ2.*prior.τ2)
end

function get_μ(prior::HS_prior)
    return zeros(prior.d)
end

function block_Gibbs_update_hyperparams(prior::HS_prior, ξ)
    #gibbs steps here 
    prior.λ2 = [rand(InverseGamma(1, 1/prior.ν[i] + ξ[i+1]^2/(2prior.τ2))) for i in 1:prior.d-1]
    prior.τ2 = rand(InverseGamma((prior.d)/2, 1/prior.γ + 0.5*sum(ξ[2:end].^2 ./ prior.λ2) ))
    prior.ν  = [rand(InverseGamma(1, 1+1/prior.λ2[i])) for i in 1:prior.d-1]
    prior.γ  = rand(InverseGamma(1, 1+1/prior.τ2))
    return prior
end

function hyperparam_size(prior::HS_prior) 
    return 2*(prior.d-1) + 2 
end

function get_hyperparams(prior::HS_prior) 
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1:prior.d-1] = prior.λ2
    hyperparams[(prior.d-1)+1] = prior.τ2
    hyperparams[(prior.d-1)+1+(1:prior.d-1)] = prior.ν
    hyperparams[(prior.d-1)+1+(prior.d-1)+1] = prior.γ
    return hyperparams
end

#--------------------------------------------------------------------------------------------------------
# Structure implementing generalised double Pareto prior
#--------------------------------------------------------------------------------------------------------

mutable struct GDP_prior <:gaussian_prior
    d ::Int64
    σ02 ::Float64 # variance for the intercept 
    # hyperparameters:
    τ ::Array{Float64}
    λ ::Array{Float64}
    α::Float64
    η::Float64
    σ2::Float64
    a::Float64
    b::Float64
end

GDP_prior(d, σ02) = GDP_prior(d, σ02, ones(d-1), ones(d-1), 1., 1., 1., 1., 1.)

function get_σ2(prior::GDP_prior)
    return vcat(prior.σ02, prior.σ2.*prior.τ)
end

function get_μ(prior::GDP_prior)
    return zeros(prior.d)
end

function block_Gibbs_update_hyperparams(prior::GDP_prior, ξ)
    σ = sqrt(prior.σ2)
    #gibbs steps here 
    prior.λ = [rand(Gamma(prior.α+1, abs(ξ[i+1])/σ + prior.η)) for i in 1:prior.d-1]
    τ_inv = [rand(InverseGaussian(prior.λ[i]*σ/abs(ξ[i+1]), prior.λ[i]^2)) for i in 1:prior.d-1]
    prior.τ = 1./τ_inv
    prior.σ2  = rand(InverseGamma(prior.a+prior.d/2, prior.b + 0.5*sum(ξ[2:end].^2 ./ prior.τ) ))
    return prior
end

function hyperparam_size(prior::GDP_prior) 
    return 2*(prior.d-1) + 1
end

function get_hyperparams(prior::GDP_prior)
    hyperparams = zeros(hyperparam_size(prior::GDP_prior))
    hyperparams[1:prior.d-1] = prior.λ
    hyperparams[(prior.d-1) + (1:prior.d-1)] = prior.τ
    hyperparams[(prior.d-1)+(prior.d-1)+1] = prior.σ2
    return hyperparams
end

function set_hyperparameters(prior::GDP_prior)
end
#--------------------------------------------------------------------------------------------------------
# Structure implementing normal gamma prior
#--------------------------------------------------------------------------------------------------------

mutable struct NG_prior <:gaussian_prior
    d ::Int64
    σ02 ::Float64 # variance for the intercept 
    # hyperparameters:
    Ψ::Array{Float64}
    λ::Float64
    γ2::Float64
    M::Float64    
end

function get_σ2(prior::NG_prior)
    return vcat(prior.σ02, prior.Ψ)
end

function get_μ(prior::NG_prior)
    return zeros(prior.d)
end

function block_Gibbs_update_hyperparams(prior::NG_prior, ξ)
    #gibbs steps here 
    for i in 1:prior.d-1 
        xi = ξ[i+1]^2
        reval("x=rgig(n=1, lambda=$(prior.λ-1/2), chi=$(1/prior.γ2), psi=$xi)")
        prior.Ψ[i] = @rget x
    end
    for i in 1:5 
        z = rand(Normal())
        λ_proposed = prior.λ*exp(z) 
        acceptance_ratio = (exp(-(λ_proposed-prior.λ))
                            *(gamma(prior.λ)/gamma(λ_proposed))^(prior.d-1)
                            *((2*prior.γ2)^(-(prior.d-1))*prod(prior.Ψ))^(λ_proposed-prior.λ) )
        if rand() < acceptance_ratio 
            prior.λ = λ_proposed
        end
    end
    prior.γ2  = 1/rand(Gamma(2+d*prior.λ, prior.M/(2*prior.λ)+sum(prior.Ψ)/2))
    return prior
end

function hyperparam_size(prior::NG_prior) 
    return (prior.d-1) + 2
end

function get_hyperparams(prior::NG_prior)
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1:prior.d-1] = prior.Ψ
    hyperparams[(prior.d-1)+1] = prior.λ
    hyperparams[(prior.d-1)+1+1] = prior.γ2
end


#--------------------------------------------------------------------------------------------------------

# For linear bounds
function get_event_time(ai::Float64, bi::Float64)     
    # this assumed that bi is non-negative
    if bi > 0 
        u = rand()
        if ai >= 0 
            return (-ai + sqrt(ai^2 - 2*bi*log(u))) / bi
        else
            return -ai/bi + sqrt(-2*log(u)/bi)
        end
    elseif bi == 0
        return rand(Exponential(1/ai))
    else 
        print("Error, slope is negative \n")
    end
end


function estimate_gradient(m::model,gw::mbsampler)
    mb = gsample(gw)
    return gradient_est, mb
end

function mbs_estimate(gw::mbsampler, f, x)
    mb = gsample(gw)
    return  sum(gw.ubf[mb].*map(f,(x[mb])))
end

#--------------------------------------------------------------------------------------------------------
# BOUNDS
#--------------------------------------------------------------------------------------------------------


linear_bound(ll::ll_model, pr::gaussian_prior, gs_list::sampler_list) = 
linear_bound(build_linear_bound(ll, pr, gs_list)..., zeros(gs_list.d), zeros(gs_list.d))   

function evaluate_bound(bb::linear_bound,t,k)
    return bb.a_fixed[k] + bb.a_xi[k] + t * (bb.b_fixed[k] + bb.b_xi[k] )  
end


pos(x) = max.(x, 0.)
    


mutable struct block_gibbs_sampler <:msampler
   λ::Float64
end

function get_event_time(mysampler::block_gibbs_sampler, mstate::zz_state, model::model)
    return rand(Exponential(1.0/mysampler.λ))
end
function evolve_path(mysampler::block_gibbs_sampler, mstate::zz_state, τ)
    mstate.ξ += τ*mstate.θ
    
end
function update_state(mysampler::block_gibbs_sampler, mstate::zz_state, model::model, τ)
    block_Gibbs_update_hyperparams(model.pr, mstate.ξ)
    return true
end


function get_event_time(mysampler::zz_sampler, mstate::zz_state, model::model)
    mysampler.bb = update_bound(mysampler.bb, model.ll, model.pr, mysampler.gs, mstate.ξ, mstate.θ)
    #-------------------------------------
    event_times = [get_event_time(mysampler.bb.a_fixed[i] + mysampler.bb.a_xi[i], mysampler.bb.b_fixed[i] + mysampler.bb.b_xi[i]) 
                      for i in 1:d]  
    τ, i0 = findmin(event_times) 
    mysampler.i0 = i0
    return τ
end

function evolve_path(mysampler::zz_sampler, mstate::zz_state, τ)
    mstate.ξ += τ*mstate.θ    
end

function update_state(mysampler::zz_sampler, mstate::zz_state, model::model, τ)
    mb = gsample(mysampler.gs.mbs[mysampler.i0])
    rate_estimated = estimate_rate(model, mstate.ξ, mstate.θ, mysampler.i0, mb, mysampler.gs)

    alpha = (rate_estimated)/evaluate_bound(mysampler.bb,τ,mysampler.i0)
    if alpha > 1
        print("alpha: ",alpha,"\n")
    end
    bounce = false
    if rand() < alpha
        mstate.θ[mysampler.i0] *= -1
        bounce = true
    end 
    return bounce
end

"""
function ZZ_sample(model::model, outp::outputscheduler)

    d, Nobs = size(model.ll.X) 
    mb_size = gs.mbs[1].mb_size

    ξ = copy(outp.opf.xi_skeleton[:,outp.opf.tcounter-1])
    θ = outp.opf.theta
    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter-1])
    
    bb = linear_bound(model.ll, model.pr, gs)
    counter = 1
    
#-------------------------------------------------------------------------
    # run sampler:
    bounce = false
    while(is_running(outp.opt))
        
        τ = get_event_time(mysampler,mstate)
        #-------------------------------------
        t += τ 
        evolve_path(mysampler,mstate,τ)
        update_state(mysampler,mstate,τ)
        
        outp = feed(outp, ξ, θ, t, bounce)
        
        counter += 1
        if counter%100_000 == 0 
            gc()
        end
    end
    #finalize(outp.opf)
    return outp
end
"""

function ZZ_block_sample(model::model, outp::outputscheduler, blocksampler::Array{msampler})

    K = length(blocksampler)
    d, Nobs = size(model.ll.X) 

    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter-1])
    
    mstate = zz_state(copy(outp.opf.xi_skeleton[:,outp.opf.tcounter-1]), copy(outp.opf.theta))
    
    #bb = linear_bound(model.ll, model.pr, gs) 
    counter = 1
    
#-------------------------------------------------------------------------
    # run sampler:
    bounce = false
    while(is_running(outp.opt))
        
        τ_list = [get_event_time(blocksampler[i],mstate, model) for i in 1:K]
        τ, k0 = findmin(τ_list)
        #-------------------------------------
        t += τ 
        
        evolve_path(blocksampler[k0], mstate, τ)
        bounce = update_state(blocksampler[k0], mstate, model, τ)
        
        #outp = feed(outp, mstate, model.pr, bounce)
        outp = feed(outp::outputscheduler, mstate::zz_state, model.pr::prior_model, t, bounce)
        
        counter += 1
        if counter%10_000 == 0 
            gc()
        end
    end
    #finalize(outp.opf)
    return outp
end



#--------------------------------------------------------------------------------------------------------
# Other stuff: 
#--------------------------------------------------------------------------------------------------------

function extract_samples(skeleton_points, bouncing_times, h) 
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
            samples[:,sample_index] = start + Δ_pos/Δ_T*(time_location - bouncing_times[i])
            time_location += h
            sample_index += 1
        end
    end
    return samples
end

function compute_configT(m::model, samples, k)
    d, Nobs = size(X) 
    n_samples = size(samples,2)
    configT = 0.0
    @showprogress for i in 1:n_samples
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


""" 
Stochastic gradient descent for finding root.
Does not work vert well 
"""
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
# GIBBS ZIG-ZAG STUFF
#--------------------------------------------------------------------------------------------


mutable struct gzz_samples
    xi_samples::Array{Float64}
    hyper_samples::Array{gaussian_prior}
    zz_nbounces::Array{Int64}
end

gzz_samples(d,T_gibbs) = gzz_samples(zeros(d,T_gibbs), Array{gaussian_prior}(T_gibbs), zeros(T_gibbs))

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
        ξ = outp.opf.xi_skeleton[:,outp.opf.tcounter-1]
        if update_hyper
            block_Gibbs_update_hyperparams(m.pr, ξ)  # this updates my_model; to check, run:
    #     print(my_model.pr, "\n")
        end
        out_samples.xi_samples[:,t] = outp.opf.xi_skeleton[:,outp.opf.tcounter-1]
        out_samples.hyper_samples[t] = deepcopy(m.pr)
        out_samples.zz_nbounces[t] = outp.opf.tcounter-1
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










