using Distributions
include("ZZ_subsampling.jl")

function ZZ_dominated_Hessian(X, y, max_time, max_attempts, β_0, θ, σ2) 

    d, Nobs = size(X) 
    
    ξ = copy(β_0)
    t, switches, attempts = 0, 0, 0
    
    Q = X*X'/4 + eye(d)./σ2
    a = [θ[i]*derivative_full(X, y, ξ, i, Nobs, σ2) for i in 1:d] 
    b = [√d*norm(Q[:,i]) for i in 1:d] 
    
    # run sampler:
    while t < max_time && attempts < max_attempts
        event_times = [get_event_time(a[i], b[i]) for i in 1:d]        
        τ, i0 = findmin(event_times)                
        t += τ 
        
        if t < max_time && attempts < max_attempts
            # attempt a bounce: 
            attempts += 1
            ξ_new = ξ + τ*θ
            θ_old = copy(θ)
            a += b*τ

            rate = pos(θ_old[i0]*derivative_full(X, y, ξ_new, i0, Nobs, σ2)[1]) 
            bound = a[i0] 
            alpha = rate/bound
            if rand() < alpha
                θ[i0] *= -1
                switches += 1
            end   
            a[i0] = θ_old[i0]*derivative_full(X, y, ξ_new, i0, Nobs, σ2)
            ξ = copy(ξ_new)
        elseif t >= max_time && attempts < max_attempts 
            # stop at end of max_time 
            ξ += (max_time-t)*θ
        end
    end
    return ξ, θ, attempts, switches
end


