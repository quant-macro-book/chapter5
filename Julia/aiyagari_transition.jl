# ======================================================= #
# Model of Aiyagari (1994)                                #
# Transition Dynamics                                     #
# Written for Textbook                                    #
# By Sagiri Kitao (Translated in Julia by Taiki Ono)      #
# ======================================================= #

# load functrions made in advance
include("tauchen.jl")

# import libraries
using Plots
using Optim
using Random
using Distributions 
using LaTeXStrings
using JLD # to save & load arrays like .mat in Matlab

@inline function main()
    
    ind_TR = 1;
        # =1 compute initial/final SS and transition 
        # =2 compute transition starting with saved initial guess
   
    
    # ====================== #
    #  SET PARAMETER VALUES  #
    # ====================== #

    mu    = 3.0;             # risk aversion (=3 baseline)             
    beta  = 0.96;            # subjective discount factor 
    delta = 0.08;            # depreciation
    alpha = 0.36;            # capital's share of income

    NT = 200;                # transition period


    # ================================================= #
    #  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY  #
    # ================================================= #

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

    tau = 0.0;           # capital income tax rate
    T = 0.0              # lump-sum transfer


    # ========================================================= #
    #  GRID FOR CAPITAL : STATE AND CONTROL : gridk and gridk2  #
    # ========================================================= #

    Nk = 300;            # number of state grids
    maxK = 20;           # maximum value of capital grid
    minK = 0.0;          # borrowing constraint(ASSUME NO BORROWING)
    curvK = 2.0;

    gridk = zeros(Nk);
    gridk[1] = minK;
    for kc in 2:Nk
        gridk[kc]=gridk[1]+(maxK-minK)*((kc-1)/(Nk-1))^curvK;
    end

    Nk2 = 800;           # number of choice grids
    gridk2 = zeros(Nk2);
    gridk2[1] = minK;
    for kc in 2:Nk2
        gridk2[kc]=gridk2[1]+(maxK-minK)*((kc-1)/(Nk2-1))^curvK;
    end


    # =================================================== #
    #  SPLIT GRID in gridk2 TO NEARBY TWO GRIDS IN gridk  #
    # =================================================== #

    kc1vec=zeros(Nk2);
    kc2vec=zeros(Nk2);

    prk1vec=zeros(Nk2);
    prk2vec=zeros(Nk2);

    for kc in 1:Nk2

        xx = gridk2[kc];

        if xx >= gridk[Nk]
  
            kc1vec[kc] = Nk;
            kc2vec[kc] = Nk;

            prk1vec[kc] = 1.0;
            prk2vec[kc] = 0.0;

        else

            ind = 1;
            while xx > gridk[ind+1]
                ind += 1;
                if ind+1 >= Nk
                    break
                end
            end

            kc1vec[kc] = ind

            if ind < Nk

                kc2vec[kc] = ind+1;
                dK=(xx-gridk[ind])/(gridk[ind+1]-gridk[ind]);
                prk1vec[kc] = 1.0-dK;
                prk2vec[kc] = dK;

            else

                kc2vec[kc] = ind;
                prk1vec[kc] = 1.0;
                prk2vec[kc] = 0.0;

            end
        end
    end


    # ================================ #
    #  INITIAL & FINAL SS COMPUTATION  #
    # ================================ #

    if ind_TR == 1
    
        for ind_SS in 1:2
        
            # capital income tax rate & initial guess of T0 & K0
            if ind_SS == 1
            
                tau = 0.0;
                # genetic
                T0 = 0.0;
                K0 = 7.459977250187922; # K_SS0 (r=0.027315593186962)

            elseif ind_SS == 2

                tau = 0.1;
                # generic
                T0 = 0.021498907022689; # T_SS1
                K0 = 7.187735402184650; # K_SS1

            end

            r0 = alpha*((K0/labor)^(alpha-1)) - delta;

            maxiterSS = 200;
            iterSS = 1;
        
            errK = 1;
            errKTol = 1e-3; 
        
            errT = 1;
            errTTol = 1e-5; 
        
            adjK = 0.05; 
            adjT = 0.1; 

            # ======================================================================== #
            # THIS IS TO AVOID USING GLOBAL VARIABLES FOR FASTER COMPUTATION IN JULIA
            vfun0 = zeros(Nl,Nk);
            mea0 = zeros(Nl,Nk);

            if ind_SS == 1

                K_SS0 = 0.0;
                T_SS0 = 0.0;
                r_SS0 = 0.0;
                mea_SS0 = zeros(Nl,Nk);

            elseif ind_SS == 2

                K_SS1 = 0.0;
                T_SS1 = 0.0;
                r_SS1 = 0.0;
                vfun_SS1 = zeros(Nl,Nk);

            end
            # ======================================================================== #

            while ((errK > errKTol) || (errT > errTTol)) && (iterSS < maxiterSS)
            
                # compute wage as a function of interest rate r0
                wage = (1-alpha)*((alpha/(r0+delta))^alpha)^(1/(1-alpha));


                # ============================= #
                #  SS VALUE FUNCTION ITERATION  #
                # ============================= #

                # initizalization
                kfunG = zeros(Nl,Nk);           # solution grid 
                kfun = similar(kfunG);          # solution level
                vfun0 = zeros(Nl,Nk);    # itinitial guess
                vfun1 = similar(kfunG);         # new value function

                errV = 10.0;
                errVTol = 1e-5;

                maxiterV = 500;
                iterV = 1;

                while (errV > errVTol) & (iterV < maxiterV)

                    @inbounds for kc in 1:Nk
                        @inbounds for lc in 1:Nl

                            vtemp = -1000000 .* ones(Nk2); 
                            kccmax = Nk2;
        
                            @inbounds for kcc in 1:Nk2 
        
                                cons = s[lc]*wage + (1+r0*(1-tau))*gridk[kc] - gridk2[kcc] + T0;
        
                                if cons <= 0.0
                                    kccmax = kcc-1; 
                                    break  
                                end
        
                                util = (cons^(1.0-mu)-1.0) / (1.0-mu);
        
                                kcc1 = Int(kc1vec[kcc]);
                                kcc2 = Int(kc2vec[kcc]);
                            
                                vpr = 0.0; 
                                for lcc in 1:Nl 

                                    vpr += prob[lc,lcc]*(prk1vec[kcc]*vfun0[lcc,kcc1] + prk2vec[kcc]*vfun0[lcc,kcc2]);
                            
                                end
        
                                vtemp[kcc] = util + beta*vpr;

                            end
             
                            t1,t2 = findmax(vtemp[1:kccmax]);
                            vfun1[lc,kc] = t1;
                            kfunG[lc,kc] = t2; # grid from gridk2
                            kfun[lc,kc] = gridk2[t2];
        
                        end
                    end
          
                    errV = maximum(abs.(vfun1-vfun0));
                    #println([iterV,errV])
                    #flush(stdout)
                    vfun0 = copy(vfun1); # update guess
                    iterV += 1
         
                end
        
                if iterV == maxiterV
                    println("WARNING!! @aiyagari_vfi2.jl VFI: iteration reached max: iter=$iterV,e rr=$errV")
                end


                # ================================ #
                #  COMPUTE INVARIANT DISTRIBUTION  #
                # ================================ #

                mea0 = ones(Nl,Nk)/(Nl*Nk); # initial guess
                mea1 = zeros(Nl,Nk);               # initialization

                errM = 1;
                errMTol = 1e-10;
                maxiterM = 10000;
                iterM = 1;

                while (errM > errMTol) && (iterM < maxiterM)

                    for kc in 1:Nk
                        for lc in 1:Nl
                        
                            kcc = Int(kfunG[lc,kc]); # from gridk2
        
                            # split to two grids in gridk
                            kcc1 = Int(kc1vec[kcc]);
                            kcc2 = Int(kc2vec[kcc]);
        
                            for lcc in 1:Nl
        
                                mea1[lcc,kcc1] += prob[lc,lcc]*prk1vec[kcc]*mea0[lc,kc];
                                mea1[lcc,kcc2] += prob[lc,lcc]*prk2vec[kcc]*mea0[lc,kc];
                            
                            end
                        end
                    end
        
                    errM = maximum(abs.(mea1-mea0));
                    mea0 = copy(mea1);
                    iterM += 1;
                    mea1 = zeros(Nl,Nk); # initialization for the next iteration
        
                end

                if iterM == maxiterM
                    println("WARN: SS iterM>maxiterM")
                    println([ind_SS, iterM, errM])
                end

                if mea0[Nk] > 0.001
                    println("WARN: mea0[Nk] LARGE")
                    println([ind_SS, mea0[Nk]])
                end


                # =========================== #
                #  COMPUTE K1 AND errK in SS  #
                # =========================== #

                K1 = sum(mea0.*kfun);
                errK = abs(K1-K0);

                # update K0 for the next iteration
                if errK > errKTol
                    K0 += adjK*(K1-K0);
                end

                # update interest rate for the next iteration
                r0 = alpha*((K0/labor)^(alpha-1))-delta;


                # ========================== #
                #  GOVERNMENT SURPLUS IN SS  #
                # ========================== #

                rev = 0.0; # tax revenue
                @inbounds for kc in 1:Nk
                    rev += sum(mea0[:,kc]) * gridk[kc] * r0 * tau;
                end

                errT = abs(rev-T0)
                if errT > errTTol
                    # update T0 for the next iteration
                    T0 += adjT*(rev-T0);
                end

                println([iterSS, K1-K0, rev-T0, K0, T0])
                flush(stdout)

                iterSS += 1;

            end

            # interest rate and capital in equilibrium(solutions)
            EQ_r = r0;
            EQ_K = K0;
            EQ_T = T0;

            println("EQ_r = $EQ_r")
            println("EQ_K = $EQ_K")
            println("EQ_T = $EQ_T")
            flush(stdout)
            
            if ind_SS == 1

                # save K0 & T0
                K_SS0 = copy(K0);
                T_SS0 = copy(T0);
                r_SS0 = copy(r0);

                # distribution
                mea_SS0 = copy(mea0);

            elseif ind_SS == 2

                # save K0 & T0
                K_SS1 = copy(K0);
                T_SS1 = copy(T0);
                r_SS1 = copy(r0);

                # value function
                vfun_SS1 = copy(vfun0);

            end
        end


        # ======================== #
        #  TRANSITION COMPUTATION  #
        # ======================== #


        # ============================== #
        #  INITIAL GUESS OF KT0 and TT0  #
        # ============================== #

        KT0 = K_SS1 .* ones(NT);
        TT0 = T_SS1 .* ones(NT);

        NT0 = 30;

        intK=(K_SS1-K_SS0)/(NT0-1);
        intT=(T_SS1-T_SS0)/(NT0-1);

        for tc=1:NT0
            KT0[tc] = K_SS0+intK*(tc-1);
            #TT0[tc] = T_SS0+intT*(tc-1);    # let T to jump to final SS value
        end

    end

    if ind_TR == 2

        iteration_saver = load("iteration_saver.jld");
        K_SS0 = copy(iteration_saver["K_SS0"]);
        K_SS1 = copy(iteration_saver["K_SS1"]);
        T_SS0 = copy(iteration_saver["T_SS0"]);
        T_SS1 = copy(iteration_saver["T_SS1"]);
        KT0 = copy(iteration_saver["KT0"]);
        TT0 = copy(iteration_saver["TT0"]);
        vfun_SS1 = copy(iteration_saver["vfun_SS1"]);
        mea_SS0 = copy(iteration_saver["mea_SS0"]);

    end


    # new capital (initialization)
    KT1 = zeros(NT);
    tau = 0.1; # raised from 0 to 0.1 at time 1


    # ================================ #
    #  rT0 BASED ON INITIAL GUESS KT0  #
    # ================================ #

    rT0 = zeros(NT);

    for tc in 1:NT
        rT0[tc]=alpha*((KT0[tc]/labor)^(alpha-1)) - delta;
    end

    gridT = 1:NT
    p1 = plot(gridT, TT0, title="TRANSFER", titlefontsize=9,legend=false)
    p2 = plot(gridT, KT0, title="K", titlefontsize=9,legend=false)
    p3 = plot(gridT, rT0, title="r", titlefontsize=9,legend=false)
    plot(p1,p2,p3,layout=(1,3))
    savefig("initial_guess.pdf")


    # policy function(initialization)
    kfunGT = zeros(NT,Nl,Nk);
    kfunT = similar(kfunGT);

    maxiterTR = 30;
    iterTR = 1;

    errK = 1.0;
    errKTol = 1e-3;

    errT = 1.0;
    errTTol = 1e-5;

    adjK = 0.05;
    adjT = 0.1;

    while ((errK > errKTol) || (errT > errTTol)) && (iterTR < maxiterTR)

        # ================================================== #
        #  COMPUTE VALUE FUNCTION FROM t=NT to 1 (BACKWARDS) #
        # ================================================== #

        vfun0 = vfun_SS1; # value in the final SS

        for tc in NT:-1:1

            r0 = rT0[tc];
            T0 = TT0[tc];
            wage = (1-alpha)*((alpha/(r0+delta))^alpha)^(1/(1-alpha));

            # initialization
            kfunG = zeros(Nl,Nk); # solution grid
            vfun1 = zeros(Nl,Nk); # new value function
            kfun = zeros(Nl,Nk);  # solution level

            @inbounds for kc in 1:Nk
                @inbounds for lc in 1:Nl

                    vtemp = -1000000 .* ones(Nk2); 
                    kccmax = Nk2;

                    @inbounds for kcc in 1:Nk2 

                        cons = s[lc]*wage + (1+r0*(1-tau))*gridk[kc] - gridk2[kcc] + T0; # NOTE: gridk2[kcc] & r0 & T0

                        if cons <= 0.0
                            kccmax = kcc-1; 
                            break  
                        end

                        util = (cons^(1.0-mu)-1.0) / (1.0-mu);

                        kcc1 = Int(kc1vec[kcc]);
                        kcc2 = Int(kc2vec[kcc]);
                    
                        vpr = 0.0; 
                        for lcc in 1:Nl 

                            vpr += prob[lc,lcc]*(prk1vec[kcc]*vfun0[lcc,kcc1] + prk2vec[kcc]*vfun0[lcc,kcc2]);
                    
                        end

                        vtemp[kcc] = util + beta*vpr;

                    end
     
                    t1,t2 = findmax(vtemp[1:kccmax]);
                    vfun1[lc,kc] = t1;
                    kfunG[lc,kc] = t2;        # solution grid from gridk2
                    kfun[lc,kc] = gridk2[t2]; # solution capital(level)

                end
            end
  
            # update vfun0 for next period (tc-1)
            vfun0 = copy(vfun1); 
            
            # save policy function (solution grid)
            kfunGT[tc,:,:] .= kfunG;

            # save capital (level)
            kfunT[tc,:,:] .= kfun;
 
        end


        # ==================================================== #
        # COMPUTE DISTRIBUTION meaT: FROM t=1 TO NT (FORWRAD)  #
        # ==================================================== #

        meaT = zeros(NT,Nl,Nk); # initialization
        meaT[1,:,:] .= copy(mea_SS0); # dist in the initial SS
        
        mea0 = mea_SS0;

        for tc in 1:NT-1

            kfunG = copy(kfunGT[tc,:,:]);
            mea1 = zeros(Nl,Nk); # initialization 

            for kc in 1:Nk
                for lc in 1:Nl
                
                    kcc = Int(kfunG[lc,kc]); # from gridk2

                    # split to two grids in gridk
                    kcc1 = Int(kc1vec[kcc]);
                    kcc2 = Int(kc2vec[kcc]);

                    for lcc in 1:Nl

                        mea1[lcc,kcc1] += prob[lc,lcc]*prk1vec[kcc]*mea0[lc,kc];
                        mea1[lcc,kcc2] += prob[lc,lcc]*prk2vec[kcc]*mea0[lc,kc];
                    
                    end
                end
            end

            meaT[tc+1,:,:] = copy(mea1);
            mea0 = copy(mea1);

        end


        # ====================== #
        #  COMPUTE KT1 AND revT  #
        # ====================== #

        errKT = zeros(NT,1);
        errTT = zeros(NT,1);

        KT1[1] = KT0[1]; # predetermined
        errKT[1] = 0.0;

        for tc in 1:NT-1

            kfun = copy(kfunT[tc,:,:]); # saving for the next period
            mea0 = meaT[tc,:,:];

            KT1[tc+1] = sum(mea0 .* kfun); #capital at the beggining of next period
            errKT[tc+1] = abs(KT1[tc]-KT0[tc]);

        end

        errK = maximum(errKT);

        # update guess KT0
        if errK > errKTol

            # KT0[1] is predetermined
            for tc in 2:NT
                KT0[tc] += adjK*(KT1[tc]-KT0[tc]);
            end
        end
        
        # government surplus
        revT = zeros(NT);
        
        for tc in 1:NT

            mea0 = copy(meaT[tc,:,:]);
            r0 = rT0[tc];

            for kc in 1:Nk
                revT[tc] += sum(mea0[:,kc]) * gridk[kc] * r0 * tau; # tax rev
            end
            
            errTT[tc] = abs(revT[tc] - TT0[tc]);

        end

        errT = maximum(abs.(errTT));

        # update guess TT0
        if errT > errTTol
            for tc in 1:NT
                TT0[tc] += adjT*(revT[tc]-TT0[tc]);
            end
        end
        
        # update rT0
        for tc in 1:NT
            rT0[tc] = alpha*((KT0[tc]/labor)^(alpha-1))-delta;
        end

        println([iterTR, errK, errT])

        iterTR += 1;

    end


    # ========== #
    #  SOLUTION  #
    # ========== #

    maxY = 100;
    norm = K_SS0;

    plot([maxY],[K_SS1/norm],color=:red,markershape=:circle,lw=2,legend=false,st=:scatter)
    plot!([1],[K_SS0/norm],color=:red,markershape=:circle,lw=2,legend=false,st=:scatter)
    plot!(gridT,KT0./norm,color=:blue,lw=2,legend=false)
    title!("Capital",titlefontsize=14)
    xlabel!("Time")
    xlims!(1,maxY)
    savefig("Fig6_aiyagari_TR_K.pdf")


    plot(gridT,TT0,color=:blue,lw=2,legend=false)
    #plot!([1],[T_SS0],color=:red,markershape=:circle,lw=2,legend=false,st=:scatter)
    plot!([maxY],[T_SS1],color=:red,markershape=:circle,lw=2,legend=false,st=:scatter)
    title!("Transfer",titlefontsize=14)
    xlabel!("Time")
    xlims!(1,maxY)
    savefig("Fig6_aiyagari_TR_T.pdf")

    
    r_SS0 = alpha*((K_SS0/labor)^(alpha-1))-delta;
    r_SS1 = alpha*((K_SS1/labor)^(alpha-1))-delta;

    norm = 100;
    plot(gridT,norm.*rT0,color=:blue,lw=2,legend=false)
    plot!([1],[norm*r_SS0],color=:red,markershape=:circle,lw=2,legend=false,st=:scatter)
    plot!([maxY],[norm*r_SS1],color=:red,markershape=:circle,legend=false,st=:scatter)
    title!("Interest Rate",titlefontsize=14)
    xlabel!("Time")
    xlims!(1,maxY)
    savefig("Fig6_aiyagari_TR_r.pdf")


    save("iteration_saver.jld","K_SS0",K_SS0,"K_SS1",K_SS1,"T_SS0",T_SS0,"T_SS1",T_SS1,"KT0",KT0,"TT0",TT0,"vfun_SS1",vfun_SS1,"mea_SS0",mea_SS0);
    save("solution.jld","K_SS0",K_SS0,"K_SS1",K_SS1,"T_SS0",T_SS0,"T_SS1",T_SS1,"KT0",KT0,"TT0",TT0,"r_SS0",r_SS0,"r_SS1",r_SS1,"rT0",rT0);

end

main()



                            



