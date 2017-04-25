function plotLARSTrajectory(beta,labs,C)
  clf;absB=sum(abs(beta),1);lstVal=max(find(absB));absB=absB(1:lstVal);
  subplot(3,1,1:2);
  for i=1:size(beta,1); 
     if ( all(beta(i,:)==0) ) continue; end; % don't plot all zero's
     strVal=max(min(find(abs(beta(i,:))))-1,1);
     plot(absB(strVal:lstVal),beta(i,strVal:lstVal),[linecol(i) '+']); 
    text(absB(lstVal),beta(i,lstVal),num2str(labs(i))); 
    hold on; 
  end
  title('\beta vs. |\beta|_1');
  subplot(3,1,3); title('SSE');
  if ( nargin > 2 )
     [Ax,H1,H2]=plotyy(absB(1:numel(C)),C,absB(1:lstVal),sum(beta(:,1:lstVal)~=0));
     legend([H1 H2],'SSE','%Active');
  else
     plot(absB(1:lstVal),sum(beta(:,1:lstVal)~=0));
  end
