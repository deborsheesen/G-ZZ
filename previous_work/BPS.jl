using Distributions
using StatsBase
using KernelDensity
include("logistic_functions.jl")
include("common_functions.jl")


function gradient_estimate(position, mb_size, X, y, c, σ_α, σ_β, include_α, replace)
    
    d, Nobs = size(X)
    α, β = position[1:Nobs], position[Nobs+1:end]
    ξ = copy(position)
    σ_ξ = sqrt((1+c^2))/σ_α
    
    grad_est = zeros(Nobs + d)
    if include_α 
        for i in 1:Nobs 
            xx, yy = X[:,i], y[i] 
            grad_est[i] = gradient_estimate_α(xx, yy, i, Nobs, ξ, c, σ_α)
        end
    end
    
    mb = sample(1:Nobs, mb_size; replace=replace) 
    X_mb = [reshape(α[mb], 1, mb_size); X[:,mb]] 
    y_mb = y[mb]
    for i in (Nobs+1):(Nobs+d)  
        grad_est[i] = sum([derivative(X_mb[:,j], y_mb[j], (i-Nobs)+1, [1; β], Nobs, σ_β) for j in 1:mb_size])/mb_size
    end
    
    return grad_est
end

function bounds_bps(position, velocity, X, y, c, σ_α, σ_β, update_α) 
    d, Nobs = size(X)
    α, β = position[1:Nobs], position[Nobs+1:end]
    ξ = copy(position)
    σ_ξ = sqrt((1+c^2)/σ_α^2)
    a, b = zeros(d+Nobs), zeros(d+Nobs)
    
    a_11, a_12 = zeros(Nobs), zeros(Nobs)    
    a_21, a_22 = zeros(d), zeros(d)
    
    if update_α
        for i in 1:Nobs 
            if i == 1
                μ_ξ = σ_ξ^2 * c * ξ[2]
            elseif i == Nobs 
                μ_ξ = σ_ξ^2 * c * ξ[Nobs-1]
            else
                μ_ξ = σ_ξ^2 * c * (ξ[i-1] + ξ[i+1])
            end
            a_11[i] = velocity[i]
            a_12[i] = velocity[i]*(α[i] - μ_ξ)/σ_ξ^2
            
            b[i] = velocity[i]/σ_ξ^2
        end
    end
    for i in (Nobs+1):(Nobs+d) 
        a_21[i-Nobs] = velocity[i]*Nobs*maximum(abs.(X[i-Nobs,:]))
        a_22[i-Nobs] = velocity[i]*β[i-Nobs]/σ_β^2
        b[i] = velocity[i]/σ_β^2
    end   
    return abs(sum(a_11)) + abs(sum(a_12)) + abs(sum(a_21)) + abs(sum(a_22)), abs(dot(velocity,b))
end

function refresh_velocity(d, Nobs, include_α)
   if include_α 
        velocity = rand(Normal(), d+Nobs) 
        velocity /= sum(velocity.^2)
    else 
        a = rand(Normal(), d)
        a /= sum(a.^2)
        velocity = [zeros(Nobs); a]
    end 
    return velocity 
end

function BPS_ss(X, y, max_attempts, max_bounces, mb_size, α_0, β_0, c, σ_α, σ_β, λ_ref=1, Print=true, include_α=true, replace=true)
    
    d, Nobs = size(X)
    velocity = refresh_velocity(d, Nobs, include_α)
    position = [α_0; β_0]
    t, bouncing_times = 0, []
    push!(bouncing_times, t)
    skeleton = zeros(d+Nobs, max_bounces+1)
    skeleton[:,1] = position
    bounces, refreshments = 0, 0
    
    attempt = 1
    while attempt < max_attempts && bounces < max_bounces

        # simulate event time 
        a, b = bounds_bps(position, velocity, X, y, c, σ_α, σ_β, include_α) 
        s = event_time_linear_bound(a, b)
        if λ_ref > 0 
            refreshment_time = rand(Exponential(1/λ_ref),1)[1]
        end
        if λ_ref == 0 || s < refreshment_time 
            t += s
            position += velocity*s

            # simulate unbiased estimator of gradient
            u = gradient_estimate(position, mb_size, X, y, c, σ_α, σ_β, include_α, replace)

            # accept or reject proposed state 
            bound = a + b*s
            switching_prob = dot(velocity,u)/bound
            if rand(1)[1] < switching_prob 
                velocity .-= 2*dot(velocity,u)/dot(u,u)*u
                push!(bouncing_times, t)
                bounces += 1
                skeleton[:,bounces+refreshments+1] = copy(position)
            end
            attempt += 1
        else
            t += refreshment_time
            push!(bouncing_times, t)
            position += velocity*refreshment_time
            velocity = refresh_velocity(d, Nobs, include_α)
            refreshments += 1
            skeleton[:,bounces+refreshments+1] = copy(position)
        end
        if refreshments + bounces > size(skeleton)[2] - 10
            skeleton_new = zeros(d+Nobs, 2*size(skeleton)[2])
            skeleton_new[:,1:size(skeleton)[2]] = copy(skeleton) 
            skeleton = copy(skeleton_new)
            skeleton_new = [] 
        end        
    end
    if Print == true 
        print(signif(100*bounces/attempt,2),"% of bounces accepted \n")
    end
    return hcat(skeleton[:,1:bounces+refreshments+1], position), push!(bouncing_times, t), refreshments, bounces
end


