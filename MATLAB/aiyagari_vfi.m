function [meank] = aiyagari_vfi1(r)

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
Nk   = 100;                     % grid size for state and control
maxK = 20;                      % maximum value of capital grid  
minK = -phi;                    % borrowing constraint
intK = (maxK-minK)/(Nk-1);      % size of capital grid increments
gridk= minK:intK:maxK;          % state of assets 

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
    
    for kc=1:Nk        
        for lc=1:Nl            
            vtemp=-1000000*ones(Nk,1);            
            for kcc=1:Nk                
                cons = s(lc)*wage + (1+r)*gridk(kc) - gridk(kcc);                
                if cons<=0
                    kccmax=kcc-1;
                    break
                end                    
                util = (cons^(1-mu)-1)/(1-mu);                                
                vpr=0;
                for lcc=1:Nl
                    vpr = vpr + prob(lc,lcc)*v(lcc,kcc); 
                end                               
                vtemp(kcc) = util + beta*vpr;                
            end % kcc
            
            [t1,t2] = max(vtemp(1:kccmax));            
            tv(lc,kc)   = t1;
            kfunG(lc,kc)= t2;
            kfun(lc,kc) = gridk(t2);            
        end % lc
    end % kc
    
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
            for lcc=1:Nl
                mea1(lcc,kcc)=mea1(lcc,kcc)+prob(lc,lcc)*mea0(lc,kc);
            end            
        end        
    end
        
    err = max(max(abs(mea1-mea0)));    
    mea0=mea1;
    iter=iter+1;
    mea1=zeros(Nl,Nk);
    
end
   
meank=sum(sum(mea0.*kfun));
   
