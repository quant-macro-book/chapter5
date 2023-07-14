function tauchen(N::Int,ρ,σ,s::Int=3)
    """
    以下のAR(1)プロセスをマルコフプロセスで近似する。
    z_t = ρz_{t-1} + ϵ_t

    利用の際にはSpecialFunctions.jlが必要。
    入力:
    nz::Int64 : グリッドポイントの数
    ρ::Float64 : AR(1)プロセスの慣性
    σ::Float64 : 撹乱項の標準偏差
    s::Int64 : グリッドの最大値と最小値を決める変数

    出力:
    z::array{float64,1} : 等間隔のグリッド
    Π::array{float64,2} : 遷移確率の行列
    Π_inv:array{float64, 2}: 定常分布
    """

    # 等間隔のグリッドを定める
    σ_z = sqrt(σ^2/(1.0-ρ^2))
    zmin = -s*σ_z
    zmax = s*σ_z
    z = collect(LinRange(zmin,zmax,N))

    # グリッド間の間隔
    w = z[2] -z[1]


    # グリッドを所与として、遷移確率を求める
    Π = zeros(N,N)

    @inbounds for row in 1:N
        # 端点から計算する
         Π[row, 1] = std_norm_cdf((z[1] - ρ*z[row] + w/2) / σ)

         Π[row, N] = 1.0- std_norm_cdf((z[N] - ρ*z[row] - w/2) / σ)

        # 内点について計算する
        @inbounds for col in 2:N-1
            Π[row, col] = (std_norm_cdf((z[col] - ρ*z[row] + w/2) / σ)
                            -std_norm_cdf((z[col] - ρ*z[row] - w/2) / σ))
        end
    end

    # 定常分布を求める
    Π_inv = zeros(N)

    # グリッドを所与として、遷移確率を求める
    dist0 = (1/N)*ones(N)
    dist1 = similar(dist0)

    # 収束に関係する設定
    maxiter = 1000
    errtol = 1e-8

    for iter in 1:maxiter
        dist1 = Π'* dist0
        err = sum(abs.(dist0- dist1))
        dist0 = copy(dist1)

        if err < errtol
            Π_inv = dist1
            break
        end

    end

    return z,Π, Π_inv
end

# 標準正規分布の累積分布
std_norm_cdf(x::Real) = 0.5 * erfc(-x/sqrt(2))
