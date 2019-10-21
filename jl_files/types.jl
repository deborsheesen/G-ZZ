# file for all abstract types

abstract type outputtimer end
abstract type outputformater end

abstract type ll_model end
abstract type prior_model end
abstract type msampler end
abstract type gsamples end
    
abstract type gaussian_prior <:prior_model end
abstract type laplace_prior <:prior_model end

abstract type bound end