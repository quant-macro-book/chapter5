function aiyagari_vfi2(p, r)

    # computes aggregate savings given aggregate interest rate r

    #  write wage as a function of interest rate
    wage = (1-p.α)*((p.α/(r+p.δ))^p.α)^(1/(1-p.α))
main(indE = 1)
    # borrowing limit
    if r<=0
        ϕ = p.b
    else
        # b is adhoc
        # the second term is natural limit
        ϕ = minimum([p.b, wage*p.s[1]/r])
    end

    # capital grid (need define in each iteration since it depends on r/phi)
    Nk     = 150                      # grid size for state
    maxK   = 20                      # maximum value of capital grid
    minK   = -ϕ                     # borrowing constraint
    Nk2    = 500                    # grid size for control

    gridk  = collect(LinRange(minK, maxK, Nk))       # state of assets
    gridk2 = collect(LinRange(minK, maxK, Nk2))
    intk = gridk[2] - gridk[1]

    #  initialize some variables
    Nl = size(p.s)[1]
    v      = zeros(Nl,Nk)
    kfun   = similar(v)
    kfunG  = zeros(Int64, Nl, Nk)
    tv     = similar(v)
    kfunG_old=similar(v)


    for VFI_iter in 1:p.maxiter

        for (ind_l, val_l) in enumerate(p.s)
            for (ind_current, val_current) in enumerate(gridk)

                ind_future_max = Nk2
                vtemp=-1000000*ones(Nk2) # initialization

                for (ind_future, val_future) in enumerate(gridk2) # control

                    cons = val_l*wage + (1+r)*val_current - val_future

                    if cons <= 0
                        ind_future_max = ind_future - 1
                        break
                    end

                    if ind_future == Nk2

                        ind_future1 = Nk
                        ind_future2 = Nk
                        pr1 = 1
                        pr2 = 0
                    else
                        temp = (ϕ + gridk2[ind_future])/intk
                        ind_future1 = Int(floor(temp) + 1)
                        ind_future2 = ind_future1 + 1
                        pr2 = (gridk2[ind_future] - gridk[ind_future1])/intk
                        pr1 = 1 - pr2
                    end

                    vpr=0
                    for ind_l_future in 1:Nl
                        vpr += p.prob[ind_l,ind_l_future]*(pr1*v[ind_l_future, ind_future1] + pr2*v[ind_l_future, ind_future2])
                    end

                    util = CRRA(cons, p.μ)
                    vtemp[ind_future] = util + p.β * vpr

                end

                t1,t2 = findmax(vtemp[1:ind_future_max])
                tv[ind_l,ind_current]   = t1
                kfunG[ind_l,ind_current]= t2
                kfun[ind_l,ind_current] = gridk2[t2]
            end
        end


        err=maximum(maximum(abs.(kfunG-kfunG_old)))
        v=copy(tv)
        kfunG_old= copy(kfunG)

        if err == 0
            break
        end
    end

    #=============
     distribution
    =============#
    mea0=ones(Nl,Nk)/(Nl*Nk)
    mea1=zeros(Nl,Nk)

    for iter in 1:p.maxiter

        for ind_current in 1:Nk
            for ind_l in 1:Nl

                ind_future = kfunG[ind_l, ind_current]

                if ind_future == Nk2
                    ind_future1 = Nk
                    ind_future2 = Nk
                    pr1 = 1
                    pr2 = 0
                else
                    temp = (ϕ + gridk2[ind_future])/intk
                    ind_future1 = Int(floor(temp) + 1)
                    ind_future2 = ind_future1 + 1
                    pr2 = (gridk2[ind_future] - gridk[ind_future1])/intk
                    pr1 = 1 - pr2
                end



                for ind_l_future in 1:Nl
                    mea1[ind_l_future, ind_future1] += p.prob[ind_l,ind_l_future]*pr1*mea0[ind_l,ind_current]
                    mea1[ind_l_future, ind_future2] += p.prob[ind_l,ind_l_future]*pr2*mea0[ind_l,ind_current]
                end
            end
        end

        err = maximum(maximum(abs.(mea1 - mea0)))
        mea0 = copy(mea1)
        mea1 = zeros(Nl, Nk)

        if err <= p.tol
            break
        end
    end

    meank= sum(mea0.*kfun)


    return meank, kfun
end
