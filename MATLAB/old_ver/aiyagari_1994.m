%==========================================================================
% Model of Aiyagari (1994)
% Written for Keizai Seminar #6
% By Sagiri Kitao
% Comments welcome --> sagiri.kitao@gmail.com
%==========================================================================

clc; close all; clear all;

global beta mu delta alpha s Nl prob b gridk kfun 

indE=4;
    % =1 plot capital demand and asset supply curves (same g-grid for state/control) 
    % =2 same as 1 but use a finer a-grid for a control 
    % =3 compute eq K and r : method 1 : search over r-grid from the bottom 
    % =4 compute eq K and r : method 2 : update new guess of r in two ways
   

% can choose a method of grid search
% (1) aiyagari_vfi1(r) : grid search over the same grid as the state (grida) 
% (2) aiyagari_vfi2(r) : grid search over a finer grid for the control (grida2) 


%==========================================================================
% SET PARAMETER VALUES
%==========================================================================

mu     = 3;               % risk aversion (=3 baseline)             
beta   = 0.96;            % subjective discount factor 
delta  = 0.08;            % depreciation
alpha  = 0.36;            % capital's share of income
b      = 3;               % borrowing limit

labor0 = 1.125709056856582;  % LABOR IN THE BASELINE (USED ONLY IN EXPERIMENTS)

%==========================================================================
% COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY
%==========================================================================

% ROUTINE tauchen.m TO COMPUTE TRANSITION MATRIX, GRID OF AN AR(1) AND
% STATIONARY DISTRIBUTION
% approximate labor endowment shocks with seven states Markov chain
% log(s_{t}) = rho*log(s_{t-1})+e_{t} 
% e_{t}~ N(0,sig^2)

Nl       = 7;             % number of discretized states
rho      = 0.6;           % first-order autoregressive coefficient
sig      = 0.4;           % intermediate value to calculate sigma (=0.4 BASE)

% prob   : transition matrix of the Markov chain
% logs   : the discretized states of log labor earnings
% invdist: the invariant distribution of Markov chain
M=2; 
[logs,prob,invdist]= tauchen(Nl,rho,sig,M);

s = exp(logs);
labor = s'*invdist;


% ADJUST PRODUCTIVITY GRID S.T. labor REMAINS UNCHANGED (IN EXPERIMENTS)
ind=0;
if ind==1
   labor;
   adj=labor0/labor;
   s=s*adj;
   labor = s'*invdist;   
end


if indE==1 | indE==2
    
    %==========================================================================
    % COMPUTE INDIVIDUAL POLICY FUNCTION AND E(a)
    %==========================================================================
    
    NR = 20;
    minR = -0.03;
    maxR = (1-beta)/beta-0.001;
    R = minR:(maxR-minR)/(NR-1):maxR;
    
    for i = 1:length(R)
        if indE==1
            A(i) = aiyagari_vfi1(R(i)); % (1) VALUE FUNCTION ITERATION (USE THE SAME GRID FOR STATE AND CONTROL)
        elseif indE==2
            A(i) = aiyagari_vfi2(R(i)); % (2) VALUE FUNCTION ITERATION (USE FINER GRID FOR CONTROL)
        end
    end
    
    %==========================================================================
    % COMPUTE K
    %==========================================================================
    
    R_K = 0:0.005:0.05;
    K   = labor*(alpha./(R_K+delta)).^(1/(1-alpha));
        
    figure('Name','Aiyagari 1994')
    xlabel('E(a) and K')
    ylabel('Interest rate')
    hold on
    plot(A,R,'r--','LineWidth',2)
    plot(A,R,'r*')
    plot(K,R_K,'b','LineWidth',2)
    xlim([0 10])
    ylim([-0.03 0.05])
    line([0,0],[-0.04,0.05])
    line([-5,10],[0 0])
    hold off
    box on
    grid on
    set(gca,'FontSize',12,'FontName','Times New Roman');
    saveas(gcf,'fig_aiyagari.eps','epsc2')
    saveas(gcf,'fig_aiyagari.pdf','pdf')
    
    
elseif indE==3
    
    %==========================================================================
    % COMPUTE K and r in EQ
    %==========================================================================
    
    rate0=0.02;    % initial guess (START WITH A VALUE LESS THAN EQ VALUE)
    adj  =0.001;
    
    ind=0;
    while ind==0
        K0=labor*(alpha./(rate0+delta)).^(1/(1-alpha));
        %K1=aiyagari_vfi1(rate0);
        K1=aiyagari_vfi2(rate0);
        if K0<K1
            ind=1;
        end
        rate0=rate0+adj;
        [ind rate0 K0 K1 K0-K1]
    end
    
    % INTEREST RATE AND CAPITAL IN EQUILIBRIUM (SOLUTIONS)
    EQ_rate=rate0
    EQ_K=K0
    
elseif indE==4
    
    %==========================================================================
    % COMPUTE K and r in EQ
    %==========================================================================
    
    rate0= 0.025; % INITIAL GUESS
    
    err=1;
    errTol=0.001;
    maxiter=200;
    iter=1;
    
    adj=0.001;
    
    while err > errTol & iter<maxiter
        
        K0=labor*(alpha/(rate0+delta))^(1/(1-alpha));
        %K1=aiyagari_vfi1(rate0);
        K1=aiyagari_vfi2(rate0);
        
        err = abs(K0-K1);
                
        % (1) UPDATE GUESS AS (r0+r(K1))/2
        %rtemp=alpha*((K1/labor)^(alpha-1))-delta;
        %rate0=(rtemp+rate0)/2;
        
        % (2) UPDATE GUESS AS r0+adj*(K0-K1)
        rate0=rate0+adj*(K0-K1);
        
        iter=iter+1;
        
    end    
    
end


if indE==3 | indE==4
    
    figure('Name','Policy function')
    hold on
    plot(gridk,kfun(1,:),'b -','LineWidth',2)
    plot(gridk,kfun(4,:),'r --','LineWidth',2)
    plot(gridk,kfun(7,:),'k -.','LineWidth',2)
    hold off
    xlabel('a')
    ylabel('a''=g(a,l) ')
    legend('l_{low}','l_{mid}','l_{high}','Location','NW')
    box on
    grid on
    xlim([-3 10])
    ylim([-3 10])
    set(gca,'FontSize',16,'FontName','Times New Roman');
    saveas(gcf,'fig_kfun.eps','epsc2')
    saveas(gcf,'fig_kfun.png','png')
    
end
