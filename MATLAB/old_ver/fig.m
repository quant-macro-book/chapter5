
%===========
% SOLUTION 
%===========

% KT0
% TT0
% rT0
norm=K_SS0;
figure('Name','Capital')
%title('Capital')
hold on
plot(1,K_SS0/norm,'o r','LineWidth',2)
plot(gridT,KT0/norm,'b','LineWidth',2)
plot(NT,K_SS1/norm,'o r','LineWidth',2)
hold off
xlabel('Time')
xlim([1 100])
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
saveas(gcf,'fig_aiyagari_TR_K.eps','epsc2')




figure('Name','Transfer')
title('Transfer')
hold on
plot(gridT,TT0,'LineWidth',2)
plot(NT,T_SS1,'o r','LineWidth',2)
hold off
xlabel('Time')
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
saveas(gcf,'fig_aiyagari_TR_T.eps','epsc2')




r_SS0=alpha*((K_SS0/labor)^(alpha-1))-delta;
r_SS1=alpha*((K_SS1/labor)^(alpha-1))-delta;

norm=100;
figure('Name','Interest Rate')
%title('Interest Rate')
hold on
plot(1,100*r_SS0,'o r','LineWidth',2)
plot(gridT,100*rT0,'b','LineWidth',2)
plot(NT,100*r_SS1,'o r','LineWidth',2)
hold off
xlabel('Time')
ylabel('%')
box on
grid on
set(gca,'Fontsize',14)
set(gca,'FontName','Times New Roman')
set(gcf,'color','w')
saveas(gcf,'fig_aiyagari_TR_r.eps','epsc2')

