include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/types.jl")

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
    ξ::Array{Float64}           # parameter
    θ::Array{Float64}           # hyper-parameter
    α::Array{Float64}           # speed of parameter
    n_bounces::Array{Int64}
    est_rate::Array{Float64}
    T::Float64
    mu::Array{Float64}          # first moment of parameter
    m2::Array{Float64}          # second moment of parameter
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

mutable struct zz_samples <:gsamples
    samples::Array{Float64}
    tcounter::Int64
    lastsamplet::Float64
    h::Float64
end

mutable struct hyp_samples <:gsamples
    samples::Array{Float64}
    tcounter::Int64
    lastsamplet::Float64
    h::Float64
end

mutable struct projopf <:outputformater
    d::Int64
    xi_skeleton::Array{Float64}
    bt_skeleton::Array{Float64}
    theta::Array{Float64} 
    alpha_skeleton::Array{Float64}
    vel_skeleton::Array{Float64}
    n_bounces::Array{Int64}
    hyper_skeleton::Array{Float64}
    tcounter::Int64
    size_increment::Int64
    A_xi
    d_out_xi::Int64
    A_hyp
    d_out_hyp::Int64
    xi_mu::Array{Float64}
    xi_m2::Array{Float64}
    xi_lastbounce::Array{Float64}
    T_lastbounce::Float64
    tot_bounces::Int64
end
projopf(A_xi, A_hyp, size_increment::Int64) = projopf(built_projopf(A_xi, A_hyp, size_increment)...)
zz_state(opf::projopf) = zz_state(opf.xi_skeleton[:,opf.tcounter], opf.theta, opf.alpha_skeleton[:,opf.tcounter], opf.n_bounces, ones(length(opf.theta)))




mutable struct maxa_opt <:outputtimer
    running::Bool
    max_attempts::Int64
    acounter::Int64
    discard::Int64
end
maxa_opt(max_attempts) = maxa_opt(true, max_attempts, 1, 0)
maxa_opt(max_attempts, discard) = maxa_opt(true, max_attempts, 1, discard)

mutable struct block_gibbs_sampler <:msampler
    λ::Float64
    nbounces::Int64
end

block_gibbs_sampler(λ) = block_gibbs_sampler(λ,0)

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





