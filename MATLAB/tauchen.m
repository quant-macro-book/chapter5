function [Z,Zprob,Zinv] = tauchen(N,rho,sigma,m)

Z     = zeros(N,1); % グリッド
Zprob = zeros(N,N); % 遷移確率の行列
Zinv  = zeros(N,1); % 定常分布

% 等間隔のグリッドを定める
% 最大値と最小値
zmax  = m*sqrt(sigma^2/(1-rho^2));
zmin  = -zmax;
% グリッド間の間隔
w = (zmax-zmin)/(N-1);

Z = linspace(zmin,zmax,N)';


% グリッドを所与として、遷移確率を求める
for j = 1:N
    for k = 1:N
        if k == 1
            Zprob(j,k) = cdf_normal((Z(1)-rho*Z(j)+w/2)/sigma);
        elseif k == N
            Zprob(j,k) = 1 - cdf_normal((Z(N)-rho*Z(j)-w/2)/sigma);
        else
            Zprob(j,k) = cdf_normal((Z(k)-rho*Z(j)+w/2)/sigma) - ...
                         cdf_normal((Z(k)-rho*Z(j)-w/2)/sigma);
        end
    end
end


% 定常分布を求める
dist0=(1/N)*ones(N,1); 
dist1=dist0; 


err = 1;
errtol = 1e-8;
iter = 1;
while err > errtol
    
    dist1= Zprob'*dist0;
    err = sum(abs(dist0-dist1));
    dist0 = dist1;
    iter = iter+1;
    [iter err];
    
end
Zinv=dist1;


% 正規分布の累積分布関数
function c = cdf_normal(x)
    c = 0.5*erfc(-x/sqrt(2));
