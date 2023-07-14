function aiyagari_vfi3(m,K)
    """
    ---------------------------------------------------------
    === Computes Aggregate Savings given Aggregate Capital K ===
    ---------------------------------------------------------
    <input>
    ・m: model structure that contains parameters
    ・K: aggregate capital
    <output>
    ・meank: aggregate savings given interest rate 
    ・kfun:  policy function
    ・gridk: asset grid(state)
    """

    r = m.alpha*((K/m.labor)^(m.alpha-1)) - m.delta;
    wage = (1-m.alpha)*((m.alpha/(r+m.delta))^m.alpha)^(1/(1-m.alpha));

    # borrowing limit
    if r <= 0.0
        phi = m.b;
    else
        phi = min(m.b,wage*m.s[1]/r);
    end

    # -phi is borrowing limit, b is adhoc
    # the second term is natural limit

    # capital grid (need define in each iteration since it depends on r/phi)
    Nk = 300;                                     # grid size for STATE 
    maxK = 20;                                    # maximum value of capital grid
    minK = -phi;                                  # borrowing constraint
    curvK = 2.0;

    gridk = zeros(Nk);
    gridk[1] = minK;
    for kc in 2:Nk
        gridk[kc]=gridk[1]+(maxK-minK)*((kc-1)/(Nk-1))^curvK;
    end



    Nk2 = 800;                                     # grid size for CONTROL
    gridk2 = zeros(Nk2);
    gridk2[1] = minK;
    for kc in 2:Nk2
        gridk2[kc]=gridk2[1]+(maxK-minK)*((kc-1)/(Nk2-1))^curvK;
    end


    #####################################################
    # SPLIT GRID in gridk2 TO NEARBY TWO GRIDS IN gridk #
    #####################################################
    
    # calculate node and weight for interpolation  
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
                ind += 1
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


    # initialize some variables
    kfunG = zeros(m.Nl,Nk);    # new index of policy function 
    kfun = similar(kfunG);     # policy function   
    v = zeros(m.Nl,Nk);        # old value function
    tv = similar(kfunG);       # new value function
    kfunG_old = zeros(m.Nl,Nk) # old policy function 

    err     = 10;   # error between old policy index and new policy index
    maxiter = 2000; # maximum number of iteration 
    iter    = 1;    # iteration counter

    while (err > 0) & (iter < maxiter)

        # tabulate the utility function such that for zero or negative
        # consumption utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        for kc in 1:Nk # k(STATE)
            for lc in 1:m.Nl # l

                kccmax = Nk2 # maximum index that satisfies c>0.0 
                vtemp = -1000000 .* ones(Nk2); # initizalization

                for kcc in 1:Nk2 # k'(CONTROL)

                    # amount of consumption given (k,l,k')
                    cons = m.s[lc]*wage + (1+r)*gridk[kc] - gridk2[kcc] 

                    if cons <= 0.0
                        # penalty for c<0.0
                        # once c becomes negative, vtemp will not be updated(=large negative number)
                        kccmax = kcc-1; 
                        break  
                    end

                    util = (cons^(1-mu)) / (1-mu);

                    # interpolation of next period's value function
                    # find node and weight for gridk2 (evaluating gridk2 in gridk) 
                    kcc1 = Int(kc1vec[kcc]);
                    kcc2 = Int(kc2vec[kcc]);
                    
                    vpr = 0.0; # next period's value function given (l,k')
                    for lcc in 1:m.Nl # expectation of next period's value function
                        
                        vpr += m.prob[lc,lcc]*(prk1vec[kcc]*v[lcc,kcc1] + prk2vec[kcc]*v[lcc,kcc2]);
                    
                    end

                    vtemp[kcc] = util + m.beta*vpr;

                end

                # find k' that  solves bellman equation
                t1,t2 = findmax(vtemp[1:kccmax]); # subject to k' achieves c>0
                tv[lc,kc] = t1;
                kfunG[lc,kc] = t2;
                kfun[lc,kc] = gridk2[t2];

            end
        end

        v = copy(tv);
        err = maximum(abs.(kfunG-kfunG_old));
        kfunG_old = copy(kfunG);
        iter += 1

    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl VFI: iteration reached max: iter=$iter,e rr=$err")
    end

    # calculate stationary distribution
    mea0=ones(m.Nl,Nk)/(m.Nl*Nk); # old distribution
    mea1=zeros(m.Nl,Nk); # new distribution
    err=1;
    errTol=0.00001;
    maxiter=2000;
    iter=1;

    while (err > errTol) & (iter < maxiter)

        for kc in 1:Nk # k
            for lc in 1:m.Nl # l
                
                kcc = Int(kfunG[lc,kc]); # index of k'(k,l)

                # interpolation of policy function 
                # split to two grids in gridk
                kcc1 = Int(kc1vec[kcc]);
                kcc2 = Int(kc2vec[kcc]);

                for lcc in 1:m.Nl # l'

                    mea1[lcc,kcc1] += m.prob[lc,lcc]*prk1vec[kcc]*mea0[lc,kc]
                    mea1[lcc,kcc2] += m.prob[lc,lcc]*prk2vec[kcc]*mea0[lc,kc]
                    
                end
            end
        end

        err = maximum(abs.(mea1-mea0));
        mea0 = copy(mea1);
        iter += 1;
        mea1 = zeros(m.Nl,Nk);

    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl INVARIANT DIST: iteration reached max: iter=$iter, err=$err")
    end

    meank = sum(sum(mea0 .* kfun));

    return meank, kfun, gridk
end