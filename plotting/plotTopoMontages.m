function []=plotTopoMontages(capfile,montages,legends)
% plot the electrode montages
[Cname,ll,xy,xyz]=readCapInf(capfile);

markersizes=[7 16 22 28];
colors     =[[1 1 1]*.5;[1 .7 .7]*.6;[.7 .7 1]*.5;[.7 1 .7*.4]]';

% make the plot
topohead(xy); hold on; 
for mi=1:numel(montages)
  si=matchstrs(montages{mi},Cname,1);
  h(mi)=plot(xy(1,si(si>0)),xy(2,si(si>0)),'ob','markersize',markersizes(mi),'linewidth',4,'color',colors(:,mi)'); 
end
text(xy(1,:),xy(2,:),Cname,'HorizontalAlignment','center','verticalalignment','middle','color',[0 0 0],'fontweight','bold')
set(gca,'visible','off');
legend(h,legends);
