## Self-contained notebook

using StatsBase, Distributions, Optim

pos(x) = max(0.0, x)

function derivative(x, y, k, ξ, Nobs, σ2) 
    return x[k]*(exp.(ξ'x)./(1+exp.(ξ'x)) - y) + ξ[k]/(Nobs*σ2[k])
end

function derivative_full(X, y, ξ, k, Nobs, σ2)  
    return sum([derivative(X[:,j], y[j], k, ξ, Nobs, σ2) for j in 1:Nobs])[1]
end

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

function extract_samples(skeleton_points, bouncing_times, h) 
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
            samples[:,sample_index] = start + Δ_pos/Δ_T*(time_location - bouncing_times[i])
            time_location += h
            sample_index += 1
        end
    end
    return samples
end

function compute_configT(samples, k, X, y, Nobs, σ2)
    d, Nobs = size(X) 
    n_samples = size(samples,2)
    configT = 0.0
    for i in 1:n_samples
        configT += samples[k,i]*derivative_full(X, y, samples[:,i], k, Nobs, σ2)
    end
    return configT/n_samples
end

function rate_CV(gradient_root, root, ξ, X, y, i0, θ, mb_size, σ2, weights=nothing, replace=true)
    d, Nobs = size(X)
    if weights == nothing 
        mb = sample(1:Nobs, mb_size; replace=replace)             
        rate_ξ = Nobs*mean([derivative(X[:,j], y[j], i0, ξ, Nobs, σ2) for j in mb])
        rate_root = Nobs*mean([derivative(X[:,j], y[j], i0, root, Nobs, σ2) for j in mb])
        rate = gradient_root[i0] + rate_ξ - rate_root
        return pos((θ[i0]*rate)[1])
    else 
        rate_1 = pos(θ[i0]*(gradient_root[i0] + (ξ[i0]-root[i0])/σ2[i0]))
        mb = wsample(1:Nobs, Weights(weights), mb_size; replace=replace)            
        rate_likelihood = mean([X[i0,j]*(exp.(ξ'X[:,j])./(1+exp.(ξ'X[:,j])) - y[j])/weights[j] for j in mb])
        rate_root = mean([X[i0,j]*(exp.(root'X[:,j])./(1+exp.(root'X[:,j])) - y[j])/weights[j] for j in mb])
        rate_combined = pos(θ[i0]*(rate_likelihood-rate_root))
        return rate_1 + rate_combined
    end  
end

function rate_iid(ξ, X, y, i0, θ, mb_size, σ2, replace=true)
    d, Nobs = size(X)
    mb = sample(1:Nobs, mb_size; replace=replace)  
    rate_temp = Nobs*mean([derivative(X[:,j], y[j], i0, ξ, Nobs, σ2) for j in mb])
    return pos((θ[i0]*rate_temp)[1]) 
end

function rate_likelihood(weights, ξ, X, y, i0, θ, mb_size, σ2, replace)
    d, Nobs = size(X)
    mb = wsample(1:Nobs, Weights(weights), mb_size; replace=replace)
    rate_temp = mean([X[i0,j]*(exp.(ξ'X[:,j])./(1+exp.(ξ'X[:,j])) - y[j])/weights[j] for j in mb])
    return pos((θ[i0]*rate_temp)[1])
end

function find_root(X, y, σ2)
    d, Nobs = size(X)
    function gradient!(F, ξ)
        F[:] = [derivative_full(X, y, ξ, i, Nobs, σ2) for i in 1:d]
    end
    neg_loglikelihood(ξ) = sum(log.(1+exp.(X'ξ)) - y.*X'ξ)    
    result = optimize(neg_loglikelihood, gradient!, zeros(d), LBFGS())
    root = result.minimizer;
end

function extend_skeleton_points(skeleton_points, extension=1000)
    m, n = size(skeleton_points)
    skeleton_new = zeros(m, n+extension)
    skeleton_new[:,1:n] = skeleton_points
    return skeleton_new
end 

function ZZ_logistic(X, y, max_time, max_attempts, β_0, θ, mb_size, root, σ2, A=nothing, control_variates=false, weights=nothing, replace=true)

    d, Nobs = size(X) 
    if A == nothing 
        A = eye(d)
    end
    m = size(A,1)
    if weights != nothing 
        weights ./= sum(weights, 2)
    end
    
    bouncing_times = []
    push!(bouncing_times, 0.)
    skeleton_points = zeros(m, 1000)
    skeleton_points[:,1] = A*copy(β_0)
    ξ = copy(β_0)
    t, switches, attempts = 0, 0, 0
    
    if control_variates == true 
        gradient_root = [derivative_full(X, y, root, i, Nobs, σ2) for i in 1:d]
        lipschitz_constants_likelihood = zeros(d, Nobs)
        for n in 1:Nobs
            lipschitz_constants_likelihood[:,n] = 1/4*[abs.(X[i,n])*norm(X[:,n]) for i in 1:d]
        end
        if weights == nothing 
            C = Nobs*(maximum(lipschitz_constants_likelihood, 2))
        else  
            bounds = maximum(lipschitz_constants_likelihood./weights, 2)
        end
    elseif control_variates == false 
        if weights == nothing 
            bounds = Nobs*(maximum(abs.(X), 2))
        else 
            bounds = maximum(abs.(X./weights),2) 
        end
    end

    # run sampler:
    while t < max_time && attempts < max_attempts
        # find next event time:
        if control_variates == true 
            if weights == nothing 
                a = [pos((θ[i]*gradient_root[i])[1]) + (C[i]+1/σ2[i])*norm(ξ-root) for i in 1:d]
                b = √d*(C + 1./σ2)
            else  
                a = [pos((θ[i]*gradient_root[i])[1]) + (bounds[i]+1/σ2[i])*norm(ξ-root) for i in 1:d]
                b = √d*(bounds + 1./σ2)
            end
        else
            a = bounds + abs.(ξ)./σ2
            b = ones(d)./σ2
        end
        event_times = [get_event_time(a[i], b[i]) for i in 1:d] 
        τ, i0 = findmin(event_times)                
        t += τ 
        if t < max_time && attempts < max_attempts
            # attempt a bounce: 
            attempts += 1
            ξ += τ*θ
            if control_variates == true && weights == nothing 
                rate = rate_CV(gradient_root, root, ξ, X, y, i0, θ, mb_size, σ2, weights, replace) 
            elseif control_variates == true && weights != nothing 
                rate = rate_CV(gradient_root, root, ξ, X, y, i0, θ, mb_size, σ2, weights[i0,:], replace)
            elseif weights == nothing
                rate = rate_iid(ξ, X, y, i0, θ, mb_size, σ2, replace)
            else
                rate_ll = rate_likelihood(weights[i0,:], ξ, X, y, i0, θ, mb_size, σ2, replace)
                rate_prior = pos(θ[i0]*ξ[i0]/σ2[i0])
                rate = rate_prior + rate_ll
            end
            bound = a[i0] + b[i0]*τ
            alpha = rate/bound
            if alpha > 1 
                print("Error, rate larger than bound \n")
                break
            elseif rand(1)[1] < alpha
                θ[i0] *= -1
                switches += 1
                skeleton_points[:,switches+1] = A*ξ
                push!(bouncing_times, t)
            end   
            if switches == size(skeleton_points,2) - 1 
                skeleton_points = extend_skeleton_points(skeleton_points)
            end
        elseif t >= max_time && attempts < max_attempts 
            # stop at end of max_time 
            ξ += (max_time-t)*θ
        end
    end
    #print(signif(100*switches/attempts,2),"% of switches accepted \n")
    return hcat(skeleton_points[:,1:switches+1], A*ξ), push!(bouncing_times, t), ξ, θ, attempts, switches
end

function acf(x, maxlag)
    n = size(x)[1]
    acf_vec = zeros(maxlag+1)
    xmean = mean(x)
    for lag in 0:maxlag
        index, index_shifted = 1:(n-lag), (lag+1):n
        acf_vec[lag+1] = mean((x[index]-xmean).*(x[index_shifted]-xmean))
    end
    acf_vec/var(x)
end

