function [meank] = aiyagari_vfi3(K)

% computes aggregate savings given aggregate CAPITAL K

global beta mu delta alpha s Nl prob b gridk kfun labor


%   compute interest rate and wage as a function of capital
r=alpha*((K/labor)^(alpha-1))-delta;     
wage=(1-alpha)*((K/labor)^alpha);
    

% borrowing limit
if r<=0
   phi = b;
else                
   phi = min(b, wage*s(1)/r);           
end

% -phi is borrowing limit, b is adhoc
% the second term is natural limit

% capital grid (need define in each iteration since it depends on r/phi)
maxK = 20;                % maximum value of capital grid  
minK = -phi;              % borrowing constraint
Nk=300; %150;                   % grid size for STATE

curvK = 2.0;

gridk=zeros(Nk,1);
gridk(1)=minK;
for kc=2:Nk
    gridk(kc)=gridk(1)+(maxK-minK)*((kc-1)/(Nk-1))^curvK;
end


Nk2=800; % 500;                        % grid size for CONTROL

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


%  initialize some variables
kfunG  = zeros(Nl,Nk);
v      = zeros(Nl,Nk); 
tv     = zeros(Nl,Nk);

kfunG_old=kfunG;

err     = 10;
maxiter = 2000;
iter    = 1;

while err >0 & iter<maxiter
    
    %  tabulate the utility function such that for zero or negative
    %  consumption utility remains a large negative number so that
    %  such values will never be chosen as utility maximizing
    
    for kc=1:Nk % STATE
        
        for lc=1:Nl
            
            vtemp  = -1000000*ones(Nk2,1); % INITIALIZATION
            kccmax = Nk2;
            
            for kcc=1:Nk2 % CONTROL
                
                cons = s(lc)*wage + (1+r)*gridk(kc) - gridk2(kcc);
                
                if cons<=0
                    kccmax=kcc-1;
                    break
                end
                    
                util = (cons^(1-mu))/(1-mu);
                               
                kcc1 = kc1vec(kcc);
                kcc2 = kc2vec(kcc);
        
                vpr=0;
                for lcc=1:Nl
                    vpr = vpr + prob(lc,lcc)*(prk1vec(kcc)*v(lcc,kcc1)+prk2vec(kcc)*v(lcc,kcc2)); 
                end
                                       
                vtemp(kcc) = util + beta*vpr;
                
            end % kcc
            
            [t1,t2] = max(vtemp(1:kccmax));
            
            tv(lc,kc)    = t1;
            kfunG(lc,kc) = t2;
            kfun(lc,kc)  = gridk2(t2);
            
        end % lc

    end % kc
    
%    err=max(max(abs(tv-v))); 
    v=tv;   
    
    err=max(max(abs(kfunG-kfunG_old)));
    kfunG_old=kfunG;     
    iter=iter+1;
    
end
   
if iter==maxiter
    disp('WARNING!! @aiyagari_vfi2.m VFI: iteration reached max: iter=',num2str(iter),' err=',num2str(err))
end



mea0=ones(Nl,Nk)/(Nl*Nk);
mea1=zeros(Nl,Nk);
err=1;
errTol=0.00001;
maxiter=2000;
iter=1;

while err > errTol & iter<maxiter

    for kc=1:Nk

        for lc=1:Nl

            kcc=kfunG(lc,kc);

            % SPLIT TO TWO GRIDS IN gridk
            kcc1 = kc1vec(kcc);
            kcc2 = kc2vec(kcc);

            for lcc=1:Nl
                mea1(lcc,kcc1) = mea1(lcc,kcc1)+prob(lc,lcc)*prk1vec(kcc)*mea0(lc,kc);
                mea1(lcc,kcc2) = mea1(lcc,kcc2)+prob(lc,lcc)*prk2vec(kcc)*mea0(lc,kc);
            end
            
        end
        
    end
        
    err = max(max(abs(mea1-mea0)));    
    mea0=mea1;
    iter=iter+1;
    mea1=zeros(Nl,Nk);
    
end
   
if iter==maxiter
    disp('WARNING!! @aiyagari_vfi2.m INVARIANT DIST: iteration reached max: iter=',num2str(iter),' err=',num2str(err))
end

meank=sum(sum(mea0.*kfun));