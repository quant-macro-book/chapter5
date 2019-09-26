using LinearAlgebra
using Parameters
using SpecialFunctions
include("tauchen.jl")

@with_kw  struct params{TF<:Float64, TI<:Int64, TV<:AbstractVector{<:Real},
                        TM<:AbstractMatrix{<:Real}}

    # parameters
    β::TF = 0.96 # subjective discount factor
    μ::TI = 3 # risk aversion (=3 baseline)
    δ::TF = 0.08 # depreciation
    α::TF = 0.36 # capital share of income
    b::TI = 3 # borrowing limit

    # labor productivity
    s::TV # labor productivity grid
    prob::TM # transition matrix
    labor::TF # aggregate labor supply

    # iteration settings
    maxiter::TI = 2000
    tol::TF = 1e-5
end

function main(;Nl::Int = 7, ρ= 0.6, σ= 0.4, M::Int = 2, indE::Int =1)

    p = get_transition_matrix(Nl, ρ, σ, M)
    @unpack α, β, δ, labor = B(4,5,6)

    if indE == 1 || indE== 2

        #############################################
        # compute individual policy function and E(a)
        #############################################

        # create rental rate grid
        NR = 20;
        minR = -0.03;
        maxR = (1.0-β)/β -0.001;
        R =  collect(LinRange(minR, maxR, NR))
        A = similar(R) #E(a)

        for (ind, R_val) in enumerate(R)
            if indE ==1
                A[ind] = aiyagari_vfi1(p,R_val)
            elseif indE == 2
                A[ind] = aiyagari_vfi2(p,R_val)
            end
        end
        R_K = collect(range(0, 0.05, step= 0.005))
        K   = labor*(α./(R_K+δ)).^(1/(1-α))

    end


end


function get_transition_matrix(Nl::Int, ρ, σ, M::Int; ind::Int=0)

    logs, trans, invdist = tauchen(Nl, ρ, σ, M)
    s = exp.(logs)
    labor = dot(s, invdist)

    #labor in the baseline (used only in only in experiments)
    labor0 = 1.125709056856582

    # adjust productivity grid s.t. labor remains unchanged(in experiments)
    if ind==1
       adj=labor0/labor;
       s=s*adj;
       labor = dot(s, invdist)
    end

    return p = params(s = s, prob = trans, labor = labor)
end
