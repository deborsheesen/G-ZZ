using Distributions, StatsBase, KernelDensity
include("logistic_functions.jl")
include("common_functions.jl")


function update_α(θ, ξ, max_time, idx, bounds_α, xx, yy, c, σ_α, Nobs)
    d = size(X)[1]
    
    σ_ξ = sqrt(1+c^2)/σ_α
    b = 1/σ_ξ^2
    bounces, abounces = 0, 0
    β = ξ[Nobs+1:end]
    
    if idx == 1 
        μ_ξ = σ_ξ^2 * c * ξ[2]
    elseif idx == Nobs 
        μ_ξ = σ_ξ^2 * c * ξ[Nobs-1]
    else
        μ_ξ = σ_ξ^2 * c * (ξ[idx-1] + ξ[idx+1])
    end
    
    s = 0.0
    while s < max_time
        
        ξ_0 = copy(ξ[idx]) 
        a = bounds_α[idx] + abs( (ξ_0 - μ_ξ)/σ_ξ^2 )  
        τ = event_time_linear_bound(a, b)
        
        if s + τ < max_time
            ξ[idx] += θ[idx]*τ
            s += τ 
            rate = pos(θ[idx]*gradient_estimate_α(xx, yy, idx, Nobs, ξ, c, σ_α))   
            bound = a + b*τ
            switching_probability = rate/bound
            if switching_probability > 1 
                print("Error. rate = ", rate, ", bound = ", bound, "\n")
            elseif rand(1)[1] < switching_probability 
                θ[idx] *= -1
                bounces += 1
            end 
            abounces += 1
        else
            ξ[idx] += θ[idx]*(max_time-s)
            s = max_time
        end
    end
    return θ, ξ, bounces, abounces
end


## I > Nobs

function update_β(θ, ξ, mb_size, max_time, idx, bounds_β, X, y, σ_β=10, replace=true) 
    
    d, Nobs = size(X)
    s = 0.0
    i0 = idx - Nobs
    α = ξ[1:Nobs]
    
    abounces = 0
    bounces = 0
    
    while s < max_time
        
        ξ_0 = copy(ξ[idx])
        a = bounds_β[i0] + abs(ξ_0)/σ_β^2
        b = 1/σ_β^2
        
        τ = event_time_linear_bound(a, b)
        
        if s + τ < max_time
            ξ[idx] += θ[idx]*τ
            s += τ 
            
            mb = sample(1:Nobs, mb_size; replace=replace) 
            X_mb = [reshape(α[mb], 1, mb_size); X[:,mb]] 
            y_mb = y[mb]
            β = ξ[Nobs+1:end]
            
            rate = θ[idx]*sum([derivative(X_mb[:,j], y_mb[j], i0+1, [1; β], Nobs, σ_β) for j in 1:mb_size])/mb_size 
            bound = bounds_β[i0] + abs(ξ_0)/σ_β^2 + τ/σ_β^2
            switching_probability = rate/bound
            
            if switching_probability > 1 
                print("Error. rate = ", rate, ", bound = ", bound, "\n")
            elseif rand(1)[1] < switching_probability 
                θ[idx] *= -1
                bounces += 1
            end 
            abounces += 1
        else
            ξ[idx] = ξ[idx] + θ[idx]*(max_time-s)
            s = max_time
        end
    end
    
    return θ, ξ, bounces, abounces
end


function NRG_ss(X, y, Nsteps, max_times, α_0, β_0, mb_size, c, σ_α, σ_β=10, modprint=10, include_α=true, replace=true)

    # define and initialise stuff:  
    d, Nobs = size(X)
    
    ξ = [α_0; β_0]
    ξ_traj = zeros(Nobs+d, Int(floor(Nsteps/modprint)))
    
    θ = ones(Nobs + d)
    bounds_α = ones(Nobs)
    bounds_β = [Nobs*maximum(abs.(X[i,:])) for i in 1:d]   
    bounces_per_iteration = zeros(Nsteps)
    abounces_per_iteration = zeros(Nsteps)
    
    # counter for 
    abounce_counter = zeros(Nobs+d) # attempted bounces in each dimension
    bounce_counter = zeros(Nobs+d) #  bounces in each dimension
    # run sampler:
    for n in 1:Nsteps
        if include_α
            idx = sample(1:Nobs+d, 1)[1]
        else
            idx = sample(1+Nobs:Nobs+d, 1)[1]
        end
        if idx <= Nobs 
            θ, ξ, bounces, abounces = update_α(θ, ξ, max_times[idx], idx, bounds_α, X[:,idx], y[idx], c, σ_α, Nobs)
        else 
            θ, ξ, bounces, abounces = update_β(θ, ξ, mb_size, max_times[idx], idx, bounds_β, X, y, σ_β, replace) 
        end
        if n % modprint == 0
            ξ_traj[:,Int(round(n/modprint))] = ξ
        end
        abounce_counter[idx] += abounces
        bounce_counter[idx] += bounces
        bounces_per_iteration[n] = bounces
        abounces_per_iteration[n] = abounces
    end
    ξ_traj, abounce_counter, bounce_counter, bounces_per_iteration, abounces_per_iteration
end


function NRG_ss_fixed_attempts(X, y, max_times, max_attempts, α_0, β_0, mb_size, c, 
                               σ_α, σ_β=10, modprint=10, include_α=true, replace=true)

    # define and initialise stuff:  
    d, Nobs = size(X)
    
    ξ = [α_0; β_0]
    ξ_traj = zeros(Nobs+d, max_attempts+1)
    ξ_traj[:,1] = ξ
    
    θ = ones(Nobs + d)
    bounds_α = ones(Nobs)
    bounds_β = [Nobs*maximum(abs.(X[i,:])) for i in 1:d]   
    
    n = 1
   
    # counter for 
    abounce_counter = zeros(Nobs+d) # attempted bounces in each dimension
    bounce_counter = zeros(Nobs+d) #  bounces in each dimension
    bounces_per_iteration = []
    abounces_per_iteration = []
    
    # run sampler:
    while sum(abounce_counter) < max_attempts
        if include_α
            idx = sample(1:Nobs+d, 1)[1]
        else
            idx = sample(1+Nobs:Nobs+d, 1)[1]
        end
        if idx <= Nobs 
            θ, ξ, bounces, abounces = update_α(θ, ξ, max_times[idx], idx, bounds_α, X[:,idx], y[idx], c, σ_α, Nobs)
        else 
            θ, ξ, bounces, abounces = update_β(θ, ξ, mb_size, max_times[idx], idx, bounds_β, X, y, σ_β, replace) 
        end
        if n+1 > size(ξ_traj)[2]
            ξ_traj_new = zeros(size(ξ_traj)[1], 2*size(ξ_traj)[2])
            ξ_traj_new[:, 1:size(ξ_traj)[2]] = copy(ξ_traj)
            ξ_traj = copy(ξ_traj_new)
            ξ_traj_new = nothing
        end
        ξ_traj[:,n+1] = ξ
        abounce_counter[idx] += abounces
        bounce_counter[idx] += bounces
        push!(bounces_per_iteration, bounces)
        push!(abounces_per_iteration, abounces)
        n += 1
        
    end
    return ξ_traj[:, modprint*(1:Int(floor((n+1)/modprint)))], abounce_counter, bounce_counter, bounces_per_iteration, abounces_per_iteration
end


