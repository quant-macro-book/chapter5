function tauchen(N,rho,sigma,m)
    """
    ---------------------------------------------------
    === AR(1)過程をtauchenの手法によって離散化する関数 ===
    ---------------------------------------------------
    ※z'= ρ*z + ε, ε~N(0,σ_{ε}^2) を離散化する

    <input>
    ・N: 離散化するグリッドの数
    ・rho: AR(1)過程の慣性(上式のρ)
    ・sigma: AR(1)過程のショック項の標準偏差(上式のσ_{ε})
    ・m: 離散化するグリッドの範囲に関するパラメータ
    <output>
    ・Z: 離散化されたグリッド
    ・Zprob: 各グリッドの遷移行列
    ・Zinv: Zの定常分布
    """
    Zprob = zeros(N,N); # 遷移確率の行列
    Zinv = zeros(N,1);  # 定常分布

    # 等間隔のグリッドを定める
    # 最大値と最小値
    zmax = m*sqrt(sigma^2/(1-rho^2));
    zmin = -zmax;
    # グリッド間の間隔
    w = (zmax-zmin)/(N-1);

    Z  = collect(range(zmin,zmax,length=N));

    # グリッド所与として遷移確率を求める
    for j in 1:N # 今期のZのインデックス
        for k in 1:N  # 来期のZのインデックス
            if k == 1
                Zprob[j,k] = cdf_normal((Z[k]-rho*Z[j]+w/2)/sigma);
            elseif k == N
                Zprob[j,k] = 1 - cdf_normal((Z[k]-rho*Z[j]-w/2)/sigma);
            else
                Zprob[j,k] = cdf_normal((Z[k]-rho*Z[j]+w/2)/sigma) - cdf_normal((Z[k]-rho*Z[j]-w/2)/sigma);
            end
        end
    end

    # 定常分布を求める
    dist0 = (1/N) .* ones(N);
    dist1 = copy(dist0);

    err = 1.0;
    errtol = 1e-8;
    iter = 1;
    while err > errtol

        dist1 = Zprob' * dist0;
        err = sum(abs.(dist0-dist1));
        dist0 = copy(dist1);
        iter += 1;

    end

    Zinv = copy(dist1);

    return Z,Zprob,Zinv

end


function cdf_normal(x)
    """
    --------------------------------
    === 標準正規分布の累積分布関数 ===
    --------------------------------
    <input>
    ・x: 
    <output>
    ・c: 標準正規分布にしたがう確率変数Xがx以下である確率
    """
    d = Normal(0,1) # 標準正規分布
    c = cdf(d,x)

    return c
    
end