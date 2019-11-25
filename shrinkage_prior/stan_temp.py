import numpy as np, pystan as ps, h5py

def run_HMC(dat, sm, n_iter, n_chains, control) :
    
    data = h5py.File(dat, "r")
    X = data["X"].value
    y = data["y"].value
    ξ_true = data["xi_true"].value
    d, Nobs = np.shape(X.transpose())
    data = dict(N=Nobs, d=d, y=y.astype(int), X=X)

    
    fit = sm.sampling(data=data, 
                      thin=1, 
                      control=control, 
                      n_jobs=4, 
                      init="random", 
                      iter=n_iter, 
                      chains=n_chains,
                      algorithm="HMC", 
                      warmup=0)


    trace = fit.extract()
    xi_samples = trace["xi"]

    cover = np.zeros(d)
    ci = np.zeros((d,2))
    for i in range(d) :
        ci[i,:] = np.percentile(xi_samples[:,i], q=[5, 95])
        cover[i] = (ci[i,0]<ξ_true[i])&(ξ_true[i]<ci[i,1])

    del fit
    
    return xi_samples, cover, ci
    
    
