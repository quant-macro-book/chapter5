
%==========================================================================
% Model of Aiyagari (1994)
% Transition Dynamics
% Written for Textbook
% By Sagiri Kitao
% Comments welcome --> sagiri.kitao@gmail.com
%==========================================================================


clc; close all; clear all;

ind_TR=2;
    % =1 COMPUTE INITIAL/FINAL SS AND TRANSITION 
    % =2 COMPUTE TRANSITION STARTING WITH SAVED INITIAL GUESS 
    
%==========================================================================
% SET PARAMETER VALUES
%==========================================================================

mu     = 3;               % RISK AVERSION           
beta   = 0.96;            % SUBJECTIVE DISCOUNT FACTOR
delta  = 0.08;            % DEPRECIATION
alpha  = 0.36;            % CAPITAL SHARE

NT=200;                   % TRANSITION PERIOD  


%==========================================================================
% COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY
%==========================================================================

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

tau      = 0;             % capital income tax rate 
T        = 0;             % lump-sum transfer


%==========================================================================
% GRID FOR CAPITAL : STATE AND CONTROL : gridk and gridk2
%==========================================================================

maxK  = 20;                      % maximum value of capital grid  
minK  = 0;                       % borrowing constraint (ASSUME NO BORROWING)

curvK = 2.0;

Nk    = 300;                     % number of state grids 

gridk=zeros(Nk,1);
gridk(1)=minK;
for kc=2:Nk
    gridk(kc)=gridk(1)+(maxK-minK)*((kc-1)/(Nk-1))^curvK;
end


Nk2 = 800; %600;                 % number of choice grids
                                                
gridk2=zeros(Nk2,1);
gridk2(1)=minK;
for kc=2:Nk2
    gridk2(kc)=gridk2(1)+(maxK-minK)*((kc-1)/(Nk2-1))^curvK;
end


%==========================================================================
% SPLIT GRID in gridk2 TO NEARBY TWO GRIDS IN gridk
%==========================================================================

kc1vec=zeros(Nk2,1);
kc2vec=zeros(Nk2,1);

prk1vec=zeros(Nk2,1);
prk2vec=zeros(Nk2,1);

for kc=1:Nk2

    xx=gridk2(kc);

    if (xx>=gridk(Nk))
        kc1vec(kc)=Nk;
        kc2vec(kc)=Nk;

        prk1vec(kc)=1;
        prk2vec(kc)=0;
    else

        ind=1;
        while xx>gridk(ind+1)
            ind=ind+1;
            if ind+1>=Nk
                break
            end
        end

        kc1vec(kc)=ind;

        if ind<Nk
            kc2vec(kc)=ind+1;

            dK=(xx-gridk(ind))/(gridk(ind+1)-gridk(ind));
            prk1vec(kc)=1-dK;
            prk2vec(kc)=dK;
        else
            kc2vec(kc)=ind;

            prk1vec(kc)=1;
            prk2vec(kc)=0;
        end

    end

end


%==========================================================================
% INITIAL & FINAL SS COMPUTATION
%==========================================================================

if ind_TR==1

for ind_SS=1:2

    % CAPITAL INCOME TAX RATE & INITIAL GUESS OF T0 & K0
    if ind_SS==1
        tau  = 0;

        % GENERIC
        T0   = 0;
        %K0   = 6;         

        K0  =  7.459977250187922; % K_SS0 (r=0.027315593186962)

    elseif ind_SS==2
        tau  = 0.1;
        
        % GENERIC
        %T0   = 0.02;  
        %K0   = 6;   

        T0 = 0.021498907022689; % T_SS1
        K0 = 7.187735402184650; % K_SS1 
    end

    r0=alpha*((K0/labor)^(alpha-1))-delta;     
    
    maxiterSS = 200;
    iterSS    = 1;
    
    errK      = 1;
    errKTol   = 1e-3; 
    
    errT      = 1;
    errTTol   = 1e-5; % 1e-4; 
    
    adjK = 0.05; 
    adjT = 0.1; 
    
    
    while (errK>errKTol | errT>errTTol) & iterSS<maxiterSS
                          
        %   COMPUTE WAGE AS A FUNCTION OF INTEREST RATE r0
        wage = (1-alpha)*((alpha/(r0+delta))^alpha)^(1/(1-alpha));         
        

        %==========================================================================
        % SS VALUE FUNCTION ITERATION
        %==========================================================================
        
        %  INITIALIZATION
        kfunG = zeros(Nl,Nk); % SOLUTION GRID
        kfun  = zeros(Nl,Nk); % SOLUTION LEVEL        
        vfun0 = zeros(Nl,Nk); % INITIAL GUESS
        vfun1 = zeros(Nl,Nk); % NEW VALUE FUNCTION
        
        errV     = 10;
        errVTol  = 1e-5;

        maxiterV = 500;
        iterV    = 1;
        
        while errV>errVTol & iterV<maxiterV
                    
            for kc=1:Nk
                
                for lc=1:Nl
                    
                    vtemp = -1000000*ones(Nk2,1);
                    kccmax = Nk2;
                    
                    for kcc=1:Nk2 
                        
                        cons = s(lc)*wage + (1+r0*(1-tau))*gridk(kc) - gridk2(kcc) + T0; 
                        
                        if cons<=0
                            kccmax=kcc-1;
                            break
                        end
                            
                        util = (cons^(1-mu)-1)/(1-mu);
                                        
                        kcc1 = kc1vec(kcc);
                        kcc2 = kc2vec(kcc);
        
                        vpr=0;
                        for lcc=1:Nl
                            vpr = vpr + prob(lc,lcc)*(prk1vec(kcc)*vfun0(lcc,kcc1)+prk2vec(kcc)*vfun0(lcc,kcc2)); 
                        end
                                       
                        vtemp(kcc) = util + beta*vpr;
                        
                    end % kcc
                    
                    [t1,t2] = max(vtemp(1:kccmax));
                    
                    vfun1(lc,kc) = t1;
                    kfunG(lc,kc) = t2; % GRID FROM gridk2
                    kfun(lc,kc)  = gridk2(t2);
                    
                end % lc
        
            end % kc
            
            errV  = max(max(abs(vfun1-vfun0))); 
            vfun0 = vfun1;  % UPDATE GUESS    
            iterV = iterV+1;
            
        end % iterV
        
        
        if iterV>maxiterV
           disp('WARN: SS iterV>maxiterV')
           ind_SS
           iterV
           r
        end
        
        
        %==========================================================================
        % COMPUTE INVARIANT DISTRIBUTION
        %==========================================================================
        
        mea0=ones(Nl,Nk)/(Nl*Nk); % INITIAL GUESS
        mea1=zeros(Nl,Nk);        % INITIALIZATION 
        
        errM=1;
        errMTol=1e-10;
        maxiterM=10000;
        iterM=1;
        
        while errM > errMTol & iterM<maxiterM
    
            for kc=1:Nk
                
                for lc=1:Nl
                    
                    kcc = kfunG(lc,kc); % FROM gridk2
                    
                    % SPLIT TO TWO GRIDS IN gridk
                    kcc1 = kc1vec(kcc);
                    kcc2 = kc2vec(kcc);
        
                    for lcc=1:Nl
                        mea1(lcc,kcc1) = mea1(lcc,kcc1)+prob(lc,lcc)*prk1vec(kcc)*mea0(lc,kc);
                        mea1(lcc,kcc2) = mea1(lcc,kcc2)+prob(lc,lcc)*prk2vec(kcc)*mea0(lc,kc);
                    end
                    
                end
                
            end
                
            errM = max(max(abs(mea1-mea0)));    
            mea0 = mea1;

            iterM = iterM+1;
            mea1  = zeros(Nl,Nk); % INITIALIZATION FOR THE NEXT ITERATION
            
        end % iterM
              
        if iterM>maxiterM
           disp('WARN: SS iterM>maxiterM')
           ind_SS
           iterM
           errM
        end
            

        if mea0(Nk)>0.0001
            disp('WARN: mea0(Nk) LARGE')
            ind_SS
            mea0(Nk)
        end
           
        
        %==========================================================================
        % COMPUTE K1 AND errK IN SS
        %==========================================================================
        
        K1 = sum(sum(mea0.*kfun)); 
        
        errK = abs(K1-K0);  
                

        % UPDATE K0 FOR THE NEXT ITERATION
        if errK>errKTol
            %adjK=0.1*rand;
            K0 = K0+adjK*(K1-K0);            
        end
        
        % UPDATE INTEREST RATE FOR THE NEXT ITERATION
        r0=alpha*((K0/labor)^(alpha-1))-delta;

        
        %==========================================================================
        % GOVERNMENT SURPLUS IN SS
        %==========================================================================
        
        rev=0; % TAX REVENUE
        for kc=1:Nk
            rev=rev+sum(mea0(:,kc))*gridk(kc)*r0*tau; 
        end        

        errT=abs(rev-T0);
        if errT>errTTol
            % UPDATE T0 FOR THE NEXT ITERATION
            T0=T0+adjT*(rev-T0); 
        end
        

        [iterSS K1-K0 rev-T0 K0 T0]
        %[iterSS errK errT]

        iterSS=iterSS+1;
        
    end % iterSS
    
    % INTEREST RATE AND CAPITAL IN EQUILIBRIUM (SOLUTIONS)
    EQ_r=r0
    EQ_K=K0
    EQ_T=T0

    if ind_SS==1

        % SAVE K0 & T0 

        K_SS0=K0;
        T_SS0=T0;

        r_SS0=r0;

        % DISTRIBUTION
        mea_SS0=mea0;

    elseif ind_SS==2

        % SAVE K0 & T0

        K_SS1=K0;
        T_SS1=T0;

        r_SS1=r0;

        % VALUE FUNCTION 
        vfun_SS1=vfun0;

    end 
    
end % ind_SS=1:2

%pause


%==========================================================================
% TRANSITION COMPUTATION
%==========================================================================


%==============================
% INITIAL GUESS OF KT0 and TT0
%==============================

KT0=K_SS1*ones(NT,1);
TT0=T_SS1*ones(NT,1);

NT0=30;

intK=(K_SS1-K_SS0)/(NT0-1);
intT=(T_SS1-T_SS0)/(NT0-1);

for tc=1:NT0
    KT0(tc)=K_SS0+intK*(tc-1);
    % TT0(tc)=T_SS0+intT*(tc-1);    % LET T JUMP TO FINAL SS VALUE
end

end % ind_TR==1


if ind_TR==2

    load('iteration_saver')

end


% NEW CAPITAL (INITIALIZATION)
KT1=zeros(NT,1);

tau=0.1; % RAISED FROM 0 TO 0.1 AT TIME 1


%================================
% rT0 BASED ON INITIAL GUESS KT0 
%================================

rT0=zeros(NT,1); % INITIALIZATION

for tc=1:NT
    rT0(tc)=alpha*((KT0(tc)/labor)^(alpha-1))-delta;
end

gridT=1:NT;

figure('Name','Initial Guess')
subplot(1,3,1)
title('TRANSFER')
hold on
plot(gridT,TT0)
hold off
box on
grid on
subplot(1,3,2)
title('K')
hold on
plot(gridT,KT0)
hold off
box on
grid on
subplot(1,3,3)
title('r')
hold on
plot(gridT,rT0)
hold off
box on
grid on



% POLICY FUNCTION (INITIALIZATION)
kfunGT=zeros(NT,Nl,Nk);

maxiterTR = 30; % 10;
iterTR    = 1;

errK     = 1;
errKTol  = 1e-3; %0.001;

errT     = 1;
errTTol  = 1e-5; %0.0001;

adjK=0.05;  
adjT=0.1;

while (errK>errKTol | errT>errTTol) & iterTR<maxiterTR

    %=====================================================
    % COMPUTE VALUE FUNCTION FROM t=NT to 1 (BACKWARDS)
    %=====================================================

    vfun0 = vfun_SS1; % VALUE IN THE FINAL SS

    for tc=NT:-1:1

        r0=rT0(tc);
        T0=TT0(tc);

        wage = (1-alpha)*((alpha/(r0+delta))^alpha)^(1/(1-alpha));

        %  INITIALIZATION
        kfunG  = zeros(Nl,Nk); % SOLUTION GRID
        vfun1  = zeros(Nl,Nk); % NEW VALUE FUNCTION
        kfun   = zeros(Nl,Nk); % SOLUTION LEVEL

        for kc=1:Nk

            for lc=1:Nl

                vtemp=-1000000*ones(Nk2,1);
                kccmax=Nk2;

                for kcc=1:Nk2 % NOTE Nk2

                    cons = s(lc)*wage + (1+r0*(1-tau))*gridk(kc) - gridk2(kcc) + T0; % NOTE: gridk2(kcc) & r0 & T0

                    if cons<=0
                        kccmax=kcc-1;
                        break
                    end

                    util = (cons^(1-mu)-1)/(1-mu);

                    kcc1=kc1vec(kcc);
                    kcc2=kc2vec(kcc);

                    vpr=0;
                    for lcc=1:Nl
                        vpr = vpr + prob(lc,lcc)*(prk1vec(kcc)*vfun0(lcc,kcc1)+prk2vec(kcc)*vfun0(lcc,kcc2));
                    end

                    vtemp(kcc) = util + beta*vpr;

                end % kcc

                [t1,t2] = max(vtemp(1:kccmax));

                vfun1(lc,kc) = t1;
                kfunG(lc,kc) = t2;           % SOLUTION GRID FROM gridk2
                kfun(lc,kc)  = gridk2(t2);   % SOLUTION CAPITAL (LEVEL)

            end % lc

        end % kc

        % UPDATE vfun0 FOR NEXT PERIOD (tc-1)
        vfun0 = vfun1;

        % SAVE POLICY FUNCTION (SOLUTION GRID)
        kfunGT(tc,:,:)=kfunG;
        
        % SAVE CAPITAL (LEVEL)
        kfunT(tc,:,:)=kfun;
        
    end % tc


    %=====================================================
    % COMPUTE DISTRIBUTION meaT: FROM t=1 TO NT (FORWARD)
    %=====================================================

    meaT=zeros(NT,Nl,Nk); % INITIALIZATION

    meaT(1,:,:)=mea_SS0; % DIST IN THE INITIAL SS

    mea0=mea_SS0; 

    for tc=1:NT-1
        
        kfunG(:,:)=kfunGT(tc,:,:);
        
        mea1=zeros(Nl,Nk); % INITIALIZATION

        for kc=1:Nk

            for lc=1:Nl
                
                kcc=kfunG(lc,kc); % FROM gridk2
                
                % SPLIT TO TWO GRIDS IN gridk
                kcc1=kc1vec(kcc);
                kcc2=kc2vec(kcc);
                    
                for lcc=1:Nl
                    mea1(lcc,kcc1)=mea1(lcc,kcc1)+prob(lc,lcc)*prk1vec(kcc)*mea0(lc,kc);
                    mea1(lcc,kcc2)=mea1(lcc,kcc2)+prob(lc,lcc)*prk2vec(kcc)*mea0(lc,kc);
                end % lcc
                
            end % lc
            
        end % kc

        meaT(tc+1,:,:)=mea1;
        
        mea0=mea1; 
        
    end % tc


    %========================================
    % COMPUTE KT1 AND revT
    %========================================
    
    errKT=zeros(NT,1);
    errTT=zeros(NT,1);

    KT1(1)=KT0(1);   % PREDETERMINED
    errKT(1)=0;

    for tc=1:NT-1
        
        kfun(:,:)=kfunT(tc,:,:); % SAVING FOR THE NEXT PERIOD
        mea0(:,:)=meaT(tc,:,:);

        KT1(tc+1)=sum(sum(mea0.*kfun)); % CAPITAL AT THE BEGINNING OF NEXT PERIOD
        
        errKT(tc+1)=abs(KT1(tc)-KT0(tc));
    end

    errK=max(errKT);

    % UPDATE GUESS KT0
    if errK>errKTol
        
        % KT0(1) IS PREDETERMINED
        for tc=2:NT
            KT0(tc)=KT0(tc)+adjK*(KT1(tc)-KT0(tc));
        end

    end

   
    % GOVERNMENT SURPLUS

    revT=zeros(NT,1);

    for tc=1:NT
        
        mea0(:,:)=meaT(tc,:,:);
        r0=rT0(tc);
        
        for kc=1:Nk
            revT(tc)=revT(tc)+sum(mea0(:,kc))*gridk(kc)*r0*tau;   % TAX REV         
        end
        
        errTT(tc)=abs(revT(tc)-TT0(tc)); % SURPLUS

    end
 
    errT=max(abs(errTT));

    % UPDATE GUESS TT0
    if errT>errTTol

        for tc=1:NT
           TT0(tc)=TT0(tc)+adjT*(revT(tc)-TT0(tc));
        end

    end       


    % UPDATE rT0
    for tc=1:NT
        rT0(tc)=alpha*((KT0(tc)/labor)^(alpha-1))-delta;
    end
    
    [iterTR errK errT]

    iterTR=iterTR+1;

end % iterTR


%===========
% SOLUTION 
%===========

% KT0
% TT0
% rT0
maxY=100;

norm=K_SS0;
figure('Name','Capital')
%title('Capital')
hold on
plot(1,K_SS0/norm,'o r','LineWidth',2)
plot(gridT,KT0/norm,'b','LineWidth',2)
plot(maxY,K_SS1/norm,'o r','LineWidth',2)
hold off
%xlabel('Time')
xlabel('期間')
xlim([1 maxY])
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
%saveas(gcf,'Fig6_aiyagari_TR_K.eps','epsc2')
saveas(gcf,'Fig6_aiyagari_TR_K.jpg','jpg')


figure('Name','Transfer')
%title('Transfer')
hold on
plot(gridT,TT0,'LineWidth',2)
plot(maxY,T_SS1,'o r','LineWidth',2)
hold off
%xlabel('Time')
xlabel('期間')
xlim([1 maxY])
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
saveas(gcf,'Fig6_aiyagari_TR_T.eps','epsc2')




r_SS0=alpha*((K_SS0/labor)^(alpha-1))-delta;
r_SS1=alpha*((K_SS1/labor)^(alpha-1))-delta;

norm=100;
figure('Name','Interest Rate')
%title('Interest Rate')
hold on
plot(1,norm*r_SS0,'o r','LineWidth',2)
plot(gridT,norm*rT0,'b','LineWidth',2)
plot(maxY,norm*r_SS1,'o r','LineWidth',2)
hold off
%xlabel('Time')
xlabel('期間')
ylabel('%')
xlim([1 maxY])
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
saveas(gcf,'Fig6_aiyagari_TR_r.eps','epsc2')



save('iteration_saver','K_SS0','K_SS1','T_SS0','T_SS1','KT0','TT0','vfun_SS1','mea_SS0')
save('solution','K_SS0','K_SS1','T_SS0','T_SS1','KT0','TT0','r_SS0','r_SS1','rT0')

save('fig_5_4.mat');

