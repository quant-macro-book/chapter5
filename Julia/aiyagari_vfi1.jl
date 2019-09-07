function aiyagari_vfi1(p::params, r)

    # computes aggregate savings given aggregate interest rate r



    #  write wage as a function of interest rate
    wage = (1-p.α)*((p.α/(r+p.δ))^p.α)^(1/(1-p.α))

    # borrowing limit
    if r<=0
        phi = p.b
    else
        # b is adhoc
        # the second term is natural limit
        phi = min(p.b, wage*p.s[1]/r)
    end

    # capital grid (need define in each iteration since it depends on r/phi)
    Nk   = 100                     # grid size for state and control
    maxK = 20                     # maximum value of capital grid
    minK = -phi                   # borrowing constraint

    gridk= collect(LinRange(minK, maxK, Nk))       # state of assets

    #  initialize some variables
    v      = zeros(p.Nl,Nk)
    kfunG  = similar(v)
    tv     = similar(v)
    kfunG_old=similar(v)


    for VFI_iter in 1:p.maxiter
        DEV = m.β*v*p.prob'
        for ind_current, val_current in enumerate(gridk)
            for ind_l, val_l in enumerate(p.s)
                vtemp=-1000000*ones(Nk)

                for ind_future, val_future in enumerate(gridk)
                    cons = val_l*wage + (1+r)*val_current - val_future

                    if cons<=0
                        ind_future_max=ind_future-1
                        break
                    else
                        util = CRRA(cons, p.mu)
                        vtemp(ind_future) = util + DEV[ind_current, ind_l]
                    end

                end

                t1,t2 = maximum(vtemp(1:kccmax))
                tv[lc,kc]   = t1
                kfunG[lc,kc]= t2
                kfun[lc,kc] = gridk[t2]
            end
        end


        err=max(max(abs(kfunG-kfunG_old)))
        if err > 0
            v=copy(tv)
            kfunG_old= copy(kfunG)
        else
            break
        end
    end

    # distribution


    return meank
end
