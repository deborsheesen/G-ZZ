using Distributions, ProgressMeter, PolyaGammaDistribution
include("zz_structures_DuLuSuSe.jl")

function PG_update(ll::ll_logistic, prior::gaussian_prior, ξ)

    d, Nobs = size(ll.X)
    κ = ll.y - 1/2;
    B_inv = Diagonal(1./get_σ2(prior))
    b = zeros(d);

    # update omega:
    ω = [rand(PolyaGamma(1,ll.X[:,i]'ξ)) for i in 1:Nobs]

    # update xi: 
    Ω = Diagonal(ω)
    V_ω = inv(ll.X*Ω*ll.X' + B_inv)
    m_ω = V_ω*(ll.X*κ+B_inv*b)
    ξ = rand(MvNormal(m_ω, (V_ω+V_ω')/2))
    
    return ξ
end

function PG_update(ll::ll_logistic_sp, prior::gaussian_prior, ξ)

    d, Nobs = size(ll.X)
    κ = ll.y - 1/2;
    B_inv = Diagonal(1./get_σ2(prior))
    b = zeros(d);

    # update omega:
    ω = [rand(PolyaGamma(1,ll.X[:,i]'ξ)) for i in 1:Nobs]

    # update xi: 
    Ω = Diagonal(ω)
    V_ω = inv(full(ll.X*Ω*ll.X' + B_inv))
    m_ω = V_ω*(ll.X*κ+B_inv*b)
    ξ = rand(MvNormal(m_ω, (V_ω+V_ω')/2))
    
    return ξ
end

function PG(model::model, ξ, iter) 
    d = length(ξ)
    xi_samples = zeros(d,iter+1)
    xi_samples[:,1] = ξ
    hyp_samples = zeros(hyperparam_size(model.pr) ,iter+1)
    hyp_samples[:,1] = get_hyperparams(model.pr)
    @showprogress for n in 1:iter 
        block_Gibbs_update_hyperparams(model.pr, ξ)
        ξ = PG_update(model.ll, model.pr, ξ)
        xi_samples[:,n+1] = ξ
        hyp_samples[:,n+1] = get_hyperparams(model.pr)
    end
    return xi_samples, hyp_samples
end
