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
abstract type gsamples end
    
# Define abstract type for gaussian prior, sub-types of this abstract types must have attributes mu and sigma2  
abstract type gaussian_prior <:prior_model end
abstract type laplace_prior <:prior_model end

abstract type bound end


mutable struct const_bound<:bound
    a::Float64
end

mutable struct linear_bound<:bound
    const_::Array{Float64} # storing whatever constants we have 
    a::Array{Float64}
    b::Array{Float64}
end

mutable struct outputscheduler
    opf::outputformater
    opt::outputtimer
end

mutable struct zz_state 
    ξ::Array{Float64}
    θ::Array{Float64}
    α::Array{Float64}
    n_bounces::Array{Int64}
    est_rate::Array{Float64}
    T::Float64
    mu::Array{Float64}
    m2::Array{Float64} #second moment
    ξ_lastbounce::Array{Float64}
    T_lastbounce::Float64
end

zz_state(d) = zz_state(zeros(d), ones(d), ones(d), zeros(d), ones(d), 0., zeros(d), zeros(d), zeros(d), 0.)


mutable struct zz_sampler <:msampler
    i0::Int64
    gs::sampler_list
    bb::bound
    L::Int64
    adapt_speed::String
end

# ---------------------------------------------------------------------------------------------------
# Functions to extract samples along the way
# ---------------------------------------------------------------------------------------------------

mutable struct zz_samples <:gsamples
    samples::Array{Float64}
    tcounter::Int64
    lastsamplet::Float64
    h::Float64
end


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

mutable struct hyp_samples <:gsamples
    samples::Array{Float64}
    tcounter::Int64
    lastsamplet::Float64
    h::Float64
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
        

function feed(outp::outputscheduler, state::zz_state, prior::prior_model, time::Float64, bounce::Bool)
    
    if add_output(outp.opf, state, time, bounce)
        outp.opf.tcounter +=1 
        if outp.opf.tcounter > size(outp.opf.bt_skeleton,2)
            outp.opf.xi_skeleton = extend_skeleton_points(outp.opf.xi_skeleton, outp.opf.size_increment)
            outp.opf.bt_skeleton = extend_skeleton_points(outp.opf.bt_skeleton, outp.opf.size_increment)
            outp.opf.hyper_skeleton = extend_skeleton_points(outp.opf.hyper_skeleton, outp.opf.size_increment)
            outp.opf.alpha_skeleton = extend_skeleton_points(outp.opf.alpha_skeleton, outp.opf.size_increment)
        end
        outp.opf.xi_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, state.ξ)
        outp.opf.bt_skeleton[:,outp.opf.tcounter] = time
        outp.opf.hyper_skeleton[:,outp.opf.tcounter] = get_hyperparameters(prior)
        outp.opf.alpha_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, state.α)
        
        outp.opf.theta = state.θ
        outp.opf.n_bounces = state.n_bounces
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
end

mutable struct projopf <:outputformater
    d::Int64
    xi_skeleton::Array{Float64}
    bt_skeleton::Array{Float64}
    theta::Array{Float64} 
    alpha_skeleton::Array{Float64}
    n_bounces::Array{Int64}
    hyper_skeleton::Array{Float64}
    hyper_size::Int64
    tcounter::Int64
    size_increment::Int64
    A
    d_out::Int64
end

projopf(A, size_increment::Int64, hyper_size::Int64) = projopf(built_projopf(A, size_increment, hyper_size)...)
projopf(A, size_increment::Int64) = projopf(built_projopf(A, size_increment, 0)...)

zz_state(opf::projopf) = zz_state(opf.xi_skeleton[:,opf.tcounter], opf.theta, opf.alpha_skeleton[:,opf.tcounter], opf.n_bounces, ones(length(opf.theta)))


function built_projopf(A, size_increment, hyper_size)
    d_out, d = size(A)
    xi_skeleton = zeros(d_out, 10*size_increment)
    bt_skeleton = zeros(1, 10*size_increment)
    tcounter = 1
    theta = ones(d)
    hyper_skeleton = ones(hyper_size, 10*size_increment)
    alpha_skeleton = ones(d_out, 10*size_increment)
    n_bounces = zeros(d)
    return d, xi_skeleton, bt_skeleton, theta, alpha_skeleton, n_bounces, hyper_skeleton, hyper_size, tcounter, size_increment, A, d_out
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

function eval_stopping(opt::maxa_opt, xi, time, bounce)
    opt.acounter+=1
    if opt.acounter >= opt.max_attempts
        opt.running = false
    end
    return opt
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

function estimate_rate(m::model, mstate::zz_state, i0, mb, gs::sampler_list)
    rate_prior = pos(mstate.θ[i0]*mstate.α[i0]*partial_derivative(m.pr, mstate.ξ, i0))
    rate_likelihood = pos(mstate.θ[i0]*mstate.α[i0]*estimate_ll_partial(m.ll, mstate.ξ, i0, mb, gs.mbs[i0]))
    return rate_prior + rate_likelihood
end

function estimate_rate(m::model, mstate::zz_state, i0, mb, gs::cvmbsampler_list)
    rate_1 = pos(mstate.θ[i0]*mstate.α[i0] * ( gs.gradient_log_posterior_root_sum[i0] + partial_derivative(m.pr, mstate.ξ, i0) - gs.gradient_log_prior_root[i0] ))
    rate_2 = pos( mstate.θ[i0]*mstate.α[i0]*m.ll.Nobs*sum((partial_derivative_vec(m.ll, mstate.ξ, i0, mb) 
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
    const_ = [maximum(abs.(ll.X[i,:]./get_weights(gs.mbs[i],1:Nobs))) for i in 1:d]
    return const_
end

function update_bound(bb::linear_bound, ll::ll_logistic, pr::gaussian_prior, gs::mbsampler_list, mstate::zz_state)
    bb.a = mstate.α .* (bb.const_ + abs.(mstate.ξ-get_μ(pr))./ get_σ2(pr))
    bb.b = mstate.α ./ get_σ2(pr)
end

#------------------------- Bounds, with control variates -----------------------------# 

function build_linear_bound(ll::ll_logistic, pr::gaussian_prior, gs::cvmbsampler_list)
    
    d, Nobs = size(ll.X)
    C_lipschitz = zeros(d, Nobs)
    for j in 1:Nobs
        C_lipschitz[:,j] = 1/4*[abs.(ll.X[i,j])*norm(ll.X[:,j]) for i in 1:d]
    end
    const_ = [maximum(C_lipschitz[i,:]./get_weights(gs.mbs[i], 1:Nobs)) for i in 1:d]
    
    return const_
end

function update_bound(bb::linear_bound, ll::ll_logistic, pr::gaussian_prior, gs::cvmbsampler_list, mstate::zz_state)
    d = length(mstate.ξ)
    norm_ = norm(gs.root-mstate.ξ)
    bb.a = pos(mstate.θ.*mstate.α.*gs.gradient_log_posterior_root_sum) + mstate.α*norm_ .* (bb.const_ + 1.0 ./get_σ2(pr))
    bb.b = mstate.α*norm(mstate.α) .* (bb.const_ + 1.0 ./get_σ2(pr))
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
# Fix this::

function build_linear_bound(ll::ll_logistic_sp, pr::gaussian_prior, gs::mbsampler_list)
    # Redefine 
    d, Nobs = size(ll.X)
    
    const_ = zeros(d)
    for i in 1:d
        nz_ind = ll.X[i,:].nzind
        const_[i] = maximum(abs.(ll.X[i,nz_ind]./get_weights(gs.mbs[i],nz_ind)))
    end
    return const_
end

function update_bound(bb::linear_bound, ll::ll_logistic_sp, pr::gaussian_prior, gs::mbsampler_list, mstate::zz_state)
    d, Nobs = size(ll.X)
    bb.a = mstate.α .* (bb.const_ + abs.(mstate.ξ-get_μ(pr))./get_σ2(pr))
    bb.b = mstate.α ./ get_σ2(pr)
end

#-------------------- Bounds, with control variates + SPARSE -----------------------------# 

function build_linear_bound(ll::ll_logistic_sp, pr::gaussian_prior, gs::cvmbsampler_list)
    
    d, Nobs = size(ll.X)
    C_lipschitz = spzeros(d, Nobs)
    const_ = zeros(d)
    normXj = [norm(ll.X[:,j]) for j in 1:Nobs]
    for i in 1:d 
        nz_ind = ll.X[i,:].nzind
        C_lipschitz[i,nz_ind] = 1/4*abs.(ll.X[i,nz_ind ]).*normXj[nz_ind]
        const_[i] = maximum( C_lipschitz[i,nz_ind]./get_weights(gs.mbs[i], nz_ind) )
    end
    
    return const_
end

function update_bound(bb::linear_bound, ll::ll_logistic_sp, pr::gaussian_prior, gs::cvmbsampler_list, mstate::zz_state)
    
    d = length(mstate.ξ)
    norm_ = norm(gs.root-mstate.ξ)
    bb.a = pos(mstate.θ.*mstate.α.*gs.gradient_log_posterior_root_sum) + mstate.α*norm_ .* (bb.const_ + 1.0 ./get_σ2(pr))
    bb.b = mstate.α*norm(mstate.α) .* (bb.const_ + 1.0 ./get_σ2(pr))
    
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

"""
To fix:
function build_linear_bound(ll::ll_zeros, pr::gaussian_prior, gs::mbsampler_list)
    d, Nobs = size(ll.X)
    a_fixed = zeros(d)
    b_fixed = zeros(d)
    return a_fixed, b_fixed
end

function update_bound(bb::linear_bound, ll::ll_zeros, pr::gaussian_prior, gs::mbsampler_list, mstate::zz_state)
    d = length(mstate.ξ)
    bb.a = abs.(mstate.ξ-get_μ(pr))./get_σ2(pr)
    bb.b = (mstate.α)./get_σ2(pr)
end
"""


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

gaussian_prior_nh(d, σ2) = gaussian_prior_nh(ones(d), σ2*ones(d))

function get_σ2(prior::gaussian_prior)
    return prior.σ2
end

function get_μ(prior::gaussian_prior)
    return prior.μ
end

function block_Gibbs_update_hyperparams(prior::gaussian_prior_nh, ξ) 
    # do nothing
end

function hyperparam_size(prior::gaussian_prior_nh) 
    return 0 
end

function get_hyperparameters(prior::gaussian_prior_nh) 
    hyperparams = zeros(hyperparam_size(prior))
    return hyperparams
end

function set_hyperparams(prior::gaussian_prior_nh, hyperparams::Array) 
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

function get_hyperparameters(prior::HS_prior) 
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1:prior.d-1] = prior.λ2
    hyperparams[(prior.d-1)+1] = prior.τ2
    hyperparams[(prior.d-1)+1+(1:prior.d-1)] = prior.ν
    hyperparams[(prior.d-1)+1+(prior.d-1)+1] = prior.γ
    return hyperparams
end

function set_hyperparams(prior::HS_prior, hyperparams::Array{Float64}) 
    @assert hyperparam_size(prior) == length(hyperparams)
    prior.λ2 = hyperparams[1:prior.d-1]
    prior.τ2 = hyperparams[(prior.d-1)+1]
    prior.ν = hyperparams[(prior.d-1)+1+(1:prior.d-1)]
    prior.γ = hyperparams[(prior.d-1)+1+(prior.d-1)+1]
end

#--------------------------------------------------------------------------------------------------------
# Structure implementing spike and slab prior
#--------------------------------------------------------------------------------------------------------

mutable struct SS_prior <:gaussian_prior
    d ::Int64
    σ02 ::Float64 # variance for the intercept 
    # hyperparameters:
    γ::Array{Int64}
    τ2::Array{Float64}
    ν::Float64
    π::Float64
    a0::Float64
    b0::Float64
    a1::Float64
    b1::Float64
    a2::Float64
    b2::Float64
end

SS_prior(d, σ02) = SS_prior(d, σ02, Int64.(ones(d-1)), ones(d-1), 1e-2, 1e-1, 1., 1., 1., 1., 1., 1.)

function get_σ2(prior::SS_prior)
    return vcat(prior.σ02, prior.γ.*prior.τ2*prior.ν + (1-prior.γ).*prior.τ2)
end

function get_μ(prior::SS_prior)
    return zeros(prior.d)
end

function block_Gibbs_update_hyperparams(prior::SS_prior, ξ)
   
    #gibbs steps here 
    p1 = prior.π/sqrt(prior.ν)*exp.(-ξ[2:end].^2./(2*prior.ν*prior.τ2))
    p2 = (1-prior.π)*exp.(-ξ[2:end].^2./(2*prior.τ2))
    p = p1./(p1+p2)
    u = rand(prior.d-1)
    prior.γ = (u.<p) + 0
    
    prior.τ2 = [prior.γ[i]*rand(InverseGamma(prior.a0+1/2,prior.b0+ξ[i+1]^2/(2*prior.ν)))+(1-prior.γ[i])*rand(InverseGamma(prior.a0+1/2,prior.b0+ξ[i+1]^2/2)) for i in 1:prior.d-1]
    prior.ν = rand(InverseGamma(prior.a1+0.5*sum(prior.γ), prior.b1+0.5*sum(prior.γ.*ξ[2:end].^2./prior.τ2)))
    prior.π = rand(Beta(prior.a2+sum(prior.γ), prior.b2+prior.d-1-sum(prior.γ)))
end

function hyperparam_size(prior::SS_prior) 
    return 2*(prior.d-1) + 2 
end

function get_hyperparameters(prior::SS_prior) 
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1:prior.d-1] = prior.γ
    hyperparams[(prior.d-1)+1:2*(prior.d-1)] = prior.τ2
    hyperparams[2*(prior.d-1)+1] = prior.ν
    hyperparams[2*(prior.d-1)+1+1] = prior.π
    return hyperparams
end

function set_hyperparams(prior::SS_prior, hyperparams::Array{Float64}) 
    @assert hyperparam_size(prior) == length(hyperparams)
    prior.γ = hyperparams[1:prior.d-1]
    prior.τ2 = hyperparams[(prior.d-1)+1:2*(prior.d-1)]
    prior.ν = hyperparams[2*(prior.d-1)+1]
    prior.π = hyperparams[2*(prior.d-1)+1+1]
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

function get_hyperparameters(prior::GDP_prior)
    hyperparams = zeros(hyperparam_size(prior::GDP_prior))
    hyperparams[1:prior.d-1] = prior.λ
    hyperparams[(prior.d-1) + (1:prior.d-1)] = prior.τ
    hyperparams[(prior.d-1)+(prior.d-1)+1] = prior.σ2
    return hyperparams
end

function set_hyperparams(prior::GDP_prior, hyperparams::Array{Float64})
    @assert hyperparam_size(prior) == length(hyperparams)
    prior.λ = hyperparams[1:prior.d-1]
    prior.τ = hyperparams[(prior.d-1) + (1:prior.d-1)]
    prior.σ2 = hyperparams[(prior.d-1)+(prior.d-1)+1]
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
    λ_scale::Float64
    λ_attempts::Int64
end

NG_prior(d, σ02) = NG_prior(d, σ02, ones(d-1), 1., 1., 1., 1., 100)

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
    for i in 1:prior.λ_attempts
        z = prior.λ_scale*rand(Normal())
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

function get_hyperparameters(prior::NG_prior)
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1:prior.d-1] = prior.Ψ
    hyperparams[(prior.d-1)+1] = prior.λ
    hyperparams[(prior.d-1)+1+1] = prior.γ2
    return hyperparams
end

function set_hyperparams(prior::NG_prior, hyperparams::Array{Float64})
    @assert hyperparam_size(prior) == length(hyperparams)
    prior.Ψ = hyperparams[1:prior.d-1]
    prior.λ = hyperparams[(prior.d-1)+1]
    prior.γ2 = hyperparams[(prior.d-1)+1+1]
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
linear_bound(build_linear_bound(ll, pr, gs_list), zeros(size(ll.X,1)), zeros(size(ll.X,1))) 

function evaluate_bound(bb::linear_bound, t, k)
    return bb.a[k] + t*bb.b[k]
end


function pos(x::Float64) 
    return max.(x, 0.)
end

function pos(x::Int64) 
    return max.(x, 0)
end

function pos(x::Array{Float64}) 
    return [pos(x[i]) for i in 1:length(x)]
end

function pos(x::Array{Int64}) 
    return [pos(x[i]) for i in 1:length(x)]
end


    


mutable struct block_gibbs_sampler <:msampler
    λ::Float64
    nbounces::Int64
end

block_gibbs_sampler(λ) = block_gibbs_sampler(λ,0)

## --------------------------------------------------------------------------------------------------
## UPDATE STEPS FOR HYPER-PARAMETERS
## --------------------------------------------------------------------------------------------------

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

## --------------------------------------------------------------------------------------------------
## UPDATE STEPS FOR PARAMETERS
## --------------------------------------------------------------------------------------------------


function get_event_time(mysampler::zz_sampler, mstate::zz_state, model::model)
    
    update_bound(mysampler.bb, model.ll, model.pr, mysampler.gs, mstate)
    a, b = mysampler.bb.a, mysampler.bb.b
    event_times = [get_event_time(a[i], b[i]) for i in 1:d]  
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
    
    alpha = (rate_estimated)/evaluate_bound(mysampler.bb, τ, mysampler.i0)
    if alpha > 1
        print("alpha: ", alpha, "\n")
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
                mstate.α /= mean(mstate.α)
            end
        elseif mysampler.adapt_speed == "by_var"  
            if minimum(mstate.m2 - mstate.mu.^2) > 0 
                mstate.α = sqrt.(mstate.m2 - mstate.mu.^2) 
                mstate.α /= mean(mstate.α)
            end
        end
    end 
    return bounce
end



function ZZ_block_sample(model::model, outp::outputscheduler, blocksampler::Array{msampler}, mstate::zz_state)

    K = length(blocksampler)
    counter = 1

    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter])
    
#-------------------------------------------------------------------------
    # run sampler:
    start = time()
    bounce = false
    while(is_running(outp.opt))
                
        τ_list = [get_event_time(blocksampler[i], mstate, model) for i in 1:K]
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
        if counter % (outp.opt.max_attempts/10) == 0 
            print(Int64(100*counter/(outp.opt.max_attempts)), "% attempts in ", round((time()-start)/60, 2), " mins \n")
        end
    end
    finalize(outp.opf)
    return outp
end


function ZZ_block_sample_discrete(model::model, opt::outputtimer, blocksampler::Array{msampler}, mstate::zz_state, xi_samples::zz_samples, pr_samples::hyp_samples)

    K = length(blocksampler)
    counter = 1
    t = 0.
    @assert xi_samples.h == pr_samples.h "Discretizations for samples and hypersamples do not match"
    
#-------------------------------------------------------------------------
    # run sampler:
    start = time()
    bounce = false
    while(is_running(opt))
                
        τ_list = [get_event_time(blocksampler[i], mstate, model) for i in 1:K]
        τ, k0 = findmin(τ_list)
        feed(xi_samples::zz_samples, mstate::zz_state, t, τ)
        feed(pr_samples::hyp_samples, prior::gaussian_prior, t, τ)
        
        t += τ 
        evolve_path(blocksampler[k0], mstate, τ)
        bounce = update_state(blocksampler[k0], mstate, model, τ)
        eval_stopping(opt::maxa_opt, mstate.ξ, t, bounce)
        
        counter += 1
        if counter%10_000 == 0 
            gc()
        end
        if counter % (opt.max_attempts/10) == 0 
            print(Int64(100*counter/(opt.max_attempts)), "% attempts in ", round((time()-start)/60, 2), " mins \n")
        end
        finalize(xi_samples)
        finalize(pr_samples)
    end
end


function ZZ_sample(model::model, outp::outputscheduler, mysampler::zz_sampler, mstate::zz_state)
    
    d, Nobs = size(model.ll.X) 
    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter])
    counter = 1
    
#-------------------------------------------------------------------------
    # run sampler:
    bounce = false
    start = time()
    while(is_running(outp.opt))
        
        τ = get_event_time(mysampler, mstate, model)
        #-------------------------------------
        t += τ 
        evolve_path(mysampler, mstate, τ)
        bounce = update_state(mysampler, mstate, model, τ)
        
        outp = feed(outp, mstate, model.pr, t, bounce)
        
        counter += 1
        if counter%100_000 == 0 
            gc()
        end
        if counter % (outp.opt.max_attempts/10) == 0 
            print(Int64(100*counter/(outp.opt.max_attempts)), "% attempts in ", round((time()-start)/60, 2), " mins \n")
        end
    end
    finalize(outp.opf)
    return outp
end




#--------------------------------------------------------------------------------------------------------
# Other stuff: 
#--------------------------------------------------------------------------------------------------------

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
# GIBBS ZIG-ZAG STUFF [OLD]
#--------------------------------------------------------------------------------------------

mutable struct gzz_state
    mzzstate::zz_state
    prior::gaussian_prior
end


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









