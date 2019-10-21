using Distributions, StatsBase, ProgressMeter
include("GenInvGaussian.jl")

#-------------------------------------------------
# Functions for PDMP's:
#-------------------------------------------------

pos(x) = max(0.0, x)

function get_event_time(ai, bi)     # for linear bounds
    # this assumed that bi is non-positive 
    if bi > 0 
        u = rand(1)[1]
        if ai >= 0 
            return (-ai + sqrt(ai^2 - 2*bi*log(u))) / bi
        else
            return -ai/bi + sqrt(-2*log(u)/bi)
        end
    elseif bi == 0
        return rand(Exponential(1/ai),1)[1]
    else 
        print("Error, slope is negative \n")
    end
end

# Define abstract type for gaussian prior, sub-types of this abstract types must have attributes mu and sigma2  
abstract type prior end
abstract type gaussian_prior <:prior end
abstract type laplace_prior <:prior end

#-------------------------------------------------
# Structure implementing Dirichlet-Laplace prior
#-------------------------------------------------
mutable struct DL_prior <: gaussian_prior
    n ::Int64
    #hyper parameters
    ψ ::Array{Float64}
    ϕ ::Array{Float64}
    τ ::Float64
    a ::Float64
end

function get_σ2(our_prior::DL_prior)
    return our_prior.ψ .* our_prior.ϕ.^2 * our_prior.τ^2
end


#-------------------------------------------------
# Method to update hyperparameters of a DL prior
#-------------------------------------------------

function block_Gibbs_update_hypparams(our_prior::DL_prior, β)
    #gibbs steps here 
    our_prior.ψ = [rand(InverseGaussian(our_prior.ϕ[j]*our_prior.τ/abs.(β[j]), 1)) for j in 1:our_prior.n]
    our_prior.τ = rand_GenInvGaussian(1-our_prior.n, 1, 2sum(abs.(β)./our_prior.ϕ))
    T = [rand_GenInvGaussian(our_prior.a-1, 1, 2abs(β[j])) for j in 1:our_prior.n]
    our_prior.ϕ = T/sum(T)
    return our_prior
end

#---------------------------------------------------
# Method to update parameters using zig-zag process 
#---------------------------------------------------

function gradient(our_prior::DL_prior, y, ξ, idx)
    σ = sqrt.(get_σ2(our_prior))
    return exp(ξ[idx])/(1+exp(ξ[idx])) - y[idx] + ξ[idx]/σ[idx]^2
end

function block_Gibbs_update_params(our_prior::DL_prior, y, ξ, θ, max_times)
    
    n = length(ξ)
    σ = sqrt.(get_σ2(our_prior))
    Bounces, ABounces = zeros(n), zeros(n)
    for idx in 1:length(ξ) 
        bounces, abounces = 0, 0
        b = 1/σ[idx]^2
        s = 0.0
        max_time = max_times[idx]
 
        while s < max_time

            ξ_0 = copy(ξ[idx]) 
            a = 1 + abs(ξ_0)/σ[idx]^2
            τ = get_event_time(a, b)

            if s + τ < max_time
                ξ[idx] += θ[idx]*τ
                s += τ 
                rate = pos(θ[idx]*gradient(our_prior, y, ξ, idx))[1] 
                bound = a + b*τ
                switching_probability = rate/bound
                if switching_probability > 1 
                    print("Error. rate = ", rate, ", bound = ", bound, "\n")
                elseif rand() < switching_probability 
                    θ[idx] *= -1
                    bounces += 1
                end 
                abounces += 1
            else
                ξ[idx] += θ[idx]*(max_time-s)
                s = max_time
            end
        end
        Bounces[idx], ABounces[idx] = bounces, abounces
    end
    return ξ, θ, Bounces, ABounces
end

#---------------------------------------------------
# Method to compute confifurational temperature
#---------------------------------------------------

function compute_configT(our_prior::DL_prior, y::Array{Float64}, samples::Array{Float64}, idx::Int64)
    n_samples = size(samples,2)
    configT = 0.0
    for i in 1:n_samples 
        configT += samples[idx,i]*gradient(our_prior, y, samples[:,i], idx)
    end
    return configT/n_samples
end



