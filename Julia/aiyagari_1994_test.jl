# ======================================================= #
# Model of Aiyagari (1994)                                #
# By Sagiri Kitao (Translated in Julia by Taiki Ono)      #
# ======================================================= #

# load functrions made in advance
include("aiyagari_vfi1.jl")
include("aiyagari_vfi2_test.jl")
include("aiyagari_vfi3.jl")
include("tauchen.jl")

# import libraries
using Plots
using Optim
using Random
using Distributions 
using LaTeXStrings

# create constructer that contains parameters
struct Model{TI<:Integer, TF<:AbstractFloat}
    
    mu::TF                # risk aversion (=3 baseline)
    beta::TF              # subjective discount factor 
    delta::TF             # depreciation
    alpha::TF             # capital's share of income
    b::TF                 # borrowing limit
    Nl::TI                # number of discretized states
    s::Array{TF,1}        # (exponentialed) discretized states of log labor earnings
    prob::Array{TF,2}     # transition matrix of the Markov chain
    labor::TF             # aggregate labor supply

end

indE = 3;
    # =1 plot capital demand and asset supply curves (same g-grid for state/control) 
    # =2 same as 1 but use a finer a-grid for a control 
    # =3 compute eq K and r : method 1 : search over r-grid from the bottom 
    # =4 compute eq K and r : method 2 : update new guess of r in two ways

    

# ===================== #
#  SET PARAMETER VALUES #
# ===================== #

mu    = 3.0;             # risk aversion (=3 baseline)             
beta  = 0.96;            # subjective discount factor 
delta = 0.08;            # depreciation
alpha = 0.36;            # capital's share of income
b     = 3.0;             # borrowing limit


# ================================================= #
#  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY  #
# ================================================= #

# ROUTINE tauchen.m TO COMPUTE TRANSITION MATRIX, GRID OF AN AR(1) AND
# STATIONARY DISTRIBUTION
# approximate labor endowment shocks with seven states Markov chain
# log(s_{t}) = rho*log(s_{t-1})+e_{t} 
# e_{t}~ N(0,sig^2)

Nl  = 7;             # number of discretized states
rho = 0.6;           # first-order autoregressive coefficient
sig = 0.4;           # intermediate value to calculate sigma (=0.4 BASE)

# prob   : transition matrix of the Markov chain
# logs   : the discretized states of log labor earnings
# invdist: the invariant distribution of Markov chain

M = 2.0;
logs,prob,invdist = tauchen(Nl,rho,sig,M);
s = exp.(logs);
labor = s'*invdist;
println(labor)

# ============================================ #
#  CREATE CONSTRUCTER THAT CONTAINS PARAMETER  #
# ============================================ #

m = Model(mu,beta,delta,alpha,b,Nl,s,prob,labor) 

aiyagari_vfi2(m,0.01)

# if (indE==1) | (indE == 2)
    
#     # ============================================= #
#     #  COMPUTE INDIVIDUAL POLICY FUNCTION AND E(a)  #
#     # ============================================= #

#     NR = 20;
#     minR = -0.03;
#     maxR = (1-m.beta)/m.beta - 0.001;
#     R = collect(range(minR,maxR,length=NR));
#     A = zeros(NR)

#     for i in 1:NR
        
#         if indE == 1
#             A[i] = aiyagari_vfi1(m,R[i])[1]
#         elseif indE == 2
#             A[i] = aiyagari_vfi2(m,R[i])[1]
#         end

#     end

#     # ========================= #
#     #  COMPUTE K (DEMAND SIDE)  #
#     # ========================= #

#     #R_K = collect(0:0.005:0.05)
#     R_K = collect(range(0,0.05,length=100))
#     K = m.labor*(m.alpha./(R_K .+ m.delta)).^(1/(1-m.alpha));

#     #plot(A,R,color=:red,linestyle=:dashdot,linewidth=2,
#     #xlabel="E(a) and K",ylabel="Interest rate",xlims=(0-0.01,10*1.01),ylims=(-0.03*1.01,0.05*1.01),legend=false)
#     plot(A,R,color=:red,linestyle=:dashdot,linewidth=2,legend=false)
#     plot!(K,R_K,color=:blue,legend=false)
#     savefig("fig_aiyagari.pdf")


# elseif indE == 3

#     # ======================== #
#     #  COMPUTE K and r in EQ   #
#     # ======================== #

#     rate0 = 0.02; # initial guess (START WITH A VALUE LESS THAN EQ VALUE)
#     adj = 0.001;

#     ind = 0 # inidicator for whether rate0 is equilibrium price 

#     while ind == 0

#         K0 = m.labor*(m.alpha/(rate0+m.delta))^(1/(1-m.alpha)); # caputal demand
#         K1 = aiyagari_vfi2(m,rate0)[1]; # asset supply

#         if K0<K1
#             global ind = 1
#         end

#         println([ind, rate0, K0, K1, K0-K1])

#         if ind == 0
#             global rate0 += adj
#         end
      
#         #println([K0,K1])

#     end

#     # INTEREST RATE AND CAPITAL IN EQUILIBRIUM (SOLUTIONS)
#     K0,kfun0,gridk0 = aiyagari_vfi2(m,rate0)
#     println([rate0,m.labor*(m.alpha/(rate0+m.delta))^(1/(1-m.alpha))])

# elseif indE == 4

#     # ======================= #
#     #  COMPUTE K and r in EQ  #
#     # ======================= #

#     K0 = 6.8; # initial guess
    
#     err = 1;
#     errTol = 0.001;
#     maxiter = 100;
#     iter = 1;
#     adj = 0.2;

#     while (err > errTol) & (iter < maxiter)
        
#         K1 = aiyagari_vfi3(m,K0)[1];

#         global err = abs(K0-K1)/K1;

#         # UPDATE GUESS AS K0+adj*(K1-K0)

#         println([iter, K0, K1, err])

#         if err > errTol
#             global K0 += adj*(K1-K0);
#             global iter += 1;
#         end

#     end

#     if iter == maxiter
#         println("WARNING!! iter=$maxiter, err=$err")
#     end

#     K0,kfun0,gridk0 = aiyagari_vfi3(m,K0)

# end


# if (indE == 3) | (indE == 4)

#     plot(gridk0,kfun0[1,:],color=:blue,linestyle=:solid,linewidth=2,label=L"l_{low}",
#     title="Policy function",xlabel=L"a",ylabel=L"a'=g(a,l)",xlims=(-3,10),ylims=(-3,10),legend=:topleft)
#     plot!(gridk0,kfun0[4,:],color=:red,linestyle=:dash,linewidth=2,label=L"l_{mid}")
#     plot!(gridk0,kfun0[7,:],color=:black,linestyle=:dashdot,linewidth=2,label=L"l_{high}")
#     savefig("fig_kfun.pdf")

# end


