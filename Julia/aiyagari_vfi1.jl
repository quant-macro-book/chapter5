function aiyagari_vfi1(m,r)
    """
    ---------------------------------------------------------
    === Computes Aggregate Savings given Interest Rate r ===
    ---------------------------------------------------------
    <input>
    ・m: model structure that contains parameters
    ・r: interest Rate
    <output>
    ・meank: aggregate savings given interest rate 
    ・kfun:  policy function
    ・gridk: asset grid
    """
    # メモ: どのパラメータを構造体の中に入れるか
    # -> beta,mu,delta,alpha,s,Nl,prob,b
    # kgridに関するパラメータは vfi1とvfi2で異なるから構造体に入れない
    
    # write wage as a function of interest rate
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
    Nk = 100;                                    # grid size for state and control
    maxK = 20;                                   # maximum value of capital grid
    minK = -phi;                                 # borrowing constraint
    gridk = collect(range(minK,maxK,length=Nk)); # state of assets

    # initialize some variables
    kfunG = zeros(m.Nl,Nk);    # new index of policy function 
    kfun = zeros(m.Nl,Nk);     # policy function   
    v = zeros(m.Nl,Nk);        # old value function
    tv = zeros(m.Nl,Nk);       # new value function
    kfunG_old = zeros(m.Nl,Nk) # old policy function 

    err     = 10;   # error between old policy index and new policy index
    maxiter = 2000; # maximum number of iteration 
    iter    = 1;    # iteration counter

    while (err > 0.0) & (iter < maxiter)

        # tabulate the utility function such that for zero or negative
        # consumption utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        for kc in 1:Nk # k
            for lc in 1:m.Nl # l

                kccmax = Nk # maximum index that satisfies c>0.0 
                vtemp = -1000000 .* ones(Nk);

                for kcc in 1:Nk # k'

                    # amount of consumption given (k,l,k')
                    cons = m.s[lc]*wage + (1+r)*gridk[kc] - gridk[kcc] 

                    if cons <= 0.0
                        # penalty for c<0.0
                        # once c becomes negative, vtemp will not be updated(=large negative number)
                        kccmax = kcc-1; 
                        break  
                    end

                    util = (cons^(1-mu)) / (1-mu);
                    
                    vpr = 0.0; # next period's value function given (l,k')
                    for lcc in 1:m.Nl # expectation of next period's value function
                        
                        vpr += m.prob[lc,lcc]*v[lcc,kcc];
                    
                    end

                    vtemp[kcc] = util + m.beta*vpr;

                end

                # find k' that  solves bellman equation
                t1,t2 = findmax(vtemp[1:kccmax]); # subject to k' achieves c>0
                tv[lc,kc] = t1;
                kfunG[lc,kc] = t2;
                kfun[lc,kc] = gridk[t2];

            end
        end

        v = copy(tv);
        err = maximum(maximum(abs.(kfunG-kfunG_old)));
        kfunG_old = copy(kfunG);
        iter += 1

    end

    # calculate stationary distribution
    mea0 = ones(m.Nl,Nk)./(m.Nl*Nk); # old distribution
    mea1 = zeros(m.Nl,Nk); # new distribution
    err = 1;
    errTol = 0.00001;
    maxiter = 2000;
    iter = 1;

    while (err > errTol) & (iter < maxiter)

        for kc in 1:Nk # k
            for lc in 1:m.Nl # l
                
                kcc = Int(kfunG[lc,kc]) # index of k'(k,l)

                for lcc in 1:m.Nl # l'

                    mea1[lcc,kcc] += m.prob[lc,lcc]*mea0[lc,kc]

                    
                end
            end
        end

        err = maximum(maximum(abs.(mea1-mea0)));
        mea0 = copy(mea1);
        iter += 1;
        mea1 = zeros(m.Nl,Nk);

    end

    meank = sum(sum(mea0 .* kfun));

    return meank, kfun, gridk
end