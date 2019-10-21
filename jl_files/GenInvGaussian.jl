#=
Code to simulate from a generalized inverse Gaussian distribution. 
Code from https://github.com/LMescheder/GenInvGaussian.jl/blob/master/src/GenInvGaussian.jl 
=#

using Distributions

function rand_GenInvGaussian(p, a, b)

    # Handle the case p < 0
    if p < 0
        p = -p
        a, b = b, a
        invert = true
    else
        invert = false
    end

    ω = sqrt(a*b)

    r = p/2 + sqrt(p*p + a*b)/2
    dproposal = rand(Gamma(r, 2/a))
    daccept = rand()

    xm = b / ( r - p) / 2

    acceptrate(x::Real) = (x./xm)^(p - r) * exp(-b/2*(1./x - 1./xm))

    while true
        sproposal = rand(Uniform(0,dproposal))
        saccept = rand(Uniform(0,daccept))

        if saccept < acceptrate(sproposal)
            return invert ? 1/sproposal : sproposal
        end
    end
end