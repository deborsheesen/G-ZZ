using Distributions
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/types.jl")
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/structs.jl")
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/priors.jl")

mutable struct model
    ll::ll_model
    pr::prior_model
end

struct ll_logistic_sp<:ll_model
    X::SparseMatrixCSC
    y::Array{Int64}
    Nobs::Int64
end

ll_logistic_sp(X, y) = ll_logistic_sp(X, y, length(y)) 

#--------------------------------------------------------------------------------------------------------
# --------------------------------- Derivative for logistic regression ----------------------------------
#--------------------------------------------------------------------------------------------------------

function log_likelihood_vec(ll::ll_logistic_sp, ξ, mb)
   return - ( log.(1 + vec(exp.(ξ'll.X[:,mb]))) - ll.y[mb] .* vec(ξ'll.X[:,mb]) )
end

function partial_derivative_vec(ll::ll_logistic_sp, ξ, k, mb) 
    mb_size = length(mb)
    nz_ind = ll.X[k,mb].nzind
    pd_vec = spzeros(mb_size)
    mb_nz_ind = mb[nz_ind]
    pd_vec[nz_ind] = ll.X[k,mb_nz_ind].* ( vec(exp.(ξ'll.X[:,mb_nz_ind]) ./ 
                                         (1+exp.(ξ'll.X[:,mb_nz_ind]))) - ll.y[mb_nz_ind] )
    return pd_vec
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


# ------------------------------------------------------------------------------------------
# ------------------------------------ GENERAL DERIVATIVES ---------------------------------
# ------------------------------------------------------------------------------------------

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

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs_list::mbsampler_list)
    return estimate_ll_partial(ll, ξ, k, mb, gs_list.mbs[k])
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::cvmbsampler_list)
    return gs.gradient_root_sum[k] +  ll.Nobs*sum((partial_derivative_vec(ll, ξ, k, mb) 
                                             - gs.gradient_root_vec[k,mb]).*get_ubf(gs.mbs[k],mb))
end

function estimate_rate(m::model, mstate::zz_state, i0, mb, gs::mbsampler_list)
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





