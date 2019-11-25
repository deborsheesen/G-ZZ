using Distributions

include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/types.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/structs.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/models.jl")
include("/home/postdoc/dsen/Desktop/G-ZZ/jl_files/priors.jl")



function build_linear_bound(ll::ll_logistic_sp, pr::gaussian_prior, gs::mbsampler_list)
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
    bb.b = mstate.α.^2 ./ get_σ2(pr)
end

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
    bb.a = pos(mstate.θ.*mstate.α.*gs.gradient_log_posterior_root_sum) + mstate.α*norm_ .* (bb.const_ + 1./get_σ2(pr))
    bb.b = mstate.α*norm(mstate.α) .* (bb.const_ + 1./get_σ2(pr))
end


linear_bound(ll::ll_model, pr::gaussian_prior, gs_list::sampler_list) = 
linear_bound(build_linear_bound(ll, pr, gs_list), zeros(size(ll.X,1)), zeros(size(ll.X,1))) 


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

function evaluate_bound(bb::linear_bound, t, k)
    return bb.a[k] + t*bb.b[k]
end


