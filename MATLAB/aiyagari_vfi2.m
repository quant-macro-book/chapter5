function [meank] = aiyagari_vfi2(r)

% computes aggregate savings given aggregate interest rate r

global beta mu delta alpha s Nl prob b gridk kfun 

%   write wage as a function of interest rate 
wage = (1-alpha)*((alpha/(r+delta))^alpha)^(1/(1-alpha));

% borrowing limit
if r<=0
   phi = b;
else
   phi = min(b, wage*s(1)/r);           
end

% -phi is borrowing limit, b is adhoc
% the second term is natural limit

% capital grid (need define in each iteration since it depends on r/phi)
maxK = 20;                      % maximum value of capital grid  
minK = -phi;                    % borrowing constraint
Nk=150;                         % grid size for STATE
intK = (maxK-minK)/(Nk-1);      % size of capital grid increments
gridk= minK:intK:maxK;          % state of assets 

Nk2=500;                        % grid size for CONTROL
intK2=(maxK-minK)/(Nk2-1); 
gridk2 = minK:intK2:maxK;


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
            
            vtemp=-1000000*ones(Nk2,1); % INITIALIZATION
            
            for kcc=1:Nk2 % CONTROL
                
                cons = s(lc)*wage + (1+r)*gridk(kc) - gridk2(kcc);
                
                if cons<=0
                    kccmax=kcc-1;
                    break
                end
                    
                util = (cons^(1-mu))/(1-mu);
                                
                if kcc==Nk2
                    kcc1=Nk;
                    kcc2=Nk;
                    pr1=1;
                    pr2=0;
                else
                    temp=(phi+gridk2(kcc))/intK;
                    kcc1=floor(temp)+1;
                    kcc2=kcc1+1;
                    pr2=(gridk2(kcc)-gridk(kcc1))/intK;
                    pr1=1-pr2;
                end
                
                vpr=0;
                for lcc=1:Nl
                    vpr = vpr + prob(lc,lcc)*(pr1*v(lcc,kcc1)+pr2*v(lcc,kcc2)); 
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
                       
                if kcc==Nk2
                    kcc1=Nk;
                    kcc2=Nk;
                    pr1=1;
                    pr2=0;
                else
                    temp=(phi+gridk2(kcc))/intK;
                    kcc1=floor(temp)+1;
                    kcc2=kcc1+1;
                    pr2=(gridk2(kcc)-gridk(kcc1))/intK;
                    pr1=1-pr2;
                end
                
            for lcc=1:Nl
                mea1(lcc,kcc1)=mea1(lcc,kcc1)+prob(lc,lcc)*pr1*mea0(lc,kc);
                mea1(lcc,kcc2)=mea1(lcc,kcc2)+prob(lc,lcc)*pr2*mea0(lc,kc);
            end
            
        end
        
    end
        
    err = max(max(abs(mea1-mea0)));    
    mea0=mea1;
    iter=iter+1;
    mea1=zeros(Nl,Nk);
    
end
   
meank=sum(sum(mea0.*kfun));
