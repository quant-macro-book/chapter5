function CRRA(cons::Real, gamma)
    """
    Compute CRRA utility function

    # Arguments
    - `cons::Real`: consumption value
    - `gamma::Real`: relative risk aversion

    # Return
    - `util::Real`: utility value
    """
    if gamma != 1.0
        util = cons^(1.0 - gamma) / (1.0 - gamma)
    else
        util = log(cons)
    end
    return util
end
