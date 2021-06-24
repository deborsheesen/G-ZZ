using Distributions
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/types.jl")
include("/home/postdoc/dsen/Desktop/codes/G-ZZ_clean/jl_files/structs.jl")

#--------------------------------------------------------------------------------------------------------
# Derivative for prior 
#--------------------------------------------------------------------------------------------------------
        
function gradient(pr::prior_model, ξ) 
    d = length(ξ)
    return [partial_derivative(pr, ξ, k) for k in 1:d]
end

function log_prior(pr::gaussian_prior, ξ) 
    return -0.5*sum((ξ - get_μ(pr)).^2 ./ get_σ2(pr))
end


function partial_derivative(pr::gaussian_prior, ξ, k) 
    return (ξ[k] - get_μ(pr)[k]) / (get_σ2(pr)[k])
end

#--------------------------------------------------------------------------------------------------------
# Structure implementing Gaussian non-hierarchical prior
#--------------------------------------------------------------------------------------------------------

mutable struct gaussian_prior_nh <:gaussian_prior
    μ::Array{Float64}
    σ2::Array{Float64}
end

gaussian_prior_nh(d, σ2) = gaussian_prior_nh(zeros(d), σ2*ones(d))

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
# Structure implementing mixed effects prior 
#--------------------------------------------------------------------------------------------------------

mutable struct MM_prior <:gaussian_prior
    d::Int64
    K::Int64
    σ2::Float64
    ϕ::Float64
    κ2::Float64
    
    a_ϕ::Float64
    b_ϕ::Float64
    a_σ::Float64
    b_σ::Float64

end

MM_prior(d, K, σ2) = MM_prior(d, K, σ2, 1., 1., 1., 1., 1., 1.)

function get_σ2(prior::MM_prior)
    return vcat(prior.κ2/prior.ϕ, [1/prior.ϕ for i in 1:prior.K], [prior.σ2 for i in 1:prior.d])
end

function get_μ(prior::MM_prior)
    d = 1+prior.K+prior.d
    return zeros(d)
end

function block_Gibbs_update_hyperparams(prior::MM_prior, ξ)
    #Gibbs steps here 
    prior.ϕ = rand(Gamma(prior.a_ϕ+(prior.K+1)/2, prior.b_ϕ+ξ[1]^2/(2*prior.κ2)+0.5*sum(ξ[2:prior.K+1].^2)))
    prior.σ2 = rand(InverseGamma(prior.a_σ+3/2, prior.b_σ+0.5*sum(ξ[1+prior.K+1:end].^2)))
end

function hyperparam_size(prior::MM_prior) 
    return 2 
end

function get_hyperparameters(prior::MM_prior) 
    hyperparams = zeros(hyperparam_size(prior))
    hyperparams[1] = prior.ϕ
    hyperparams[2] = prior.σ2
    return hyperparams
end

function set_hyperparams(prior::MM_prior, hyperparams::Array{Float64}) 
    @assert hyperparam_size(prior) == length(hyperparams)
    prior.ϕ = hyperparams[1]
    prior.σ2 = hyperparams[2]
end
