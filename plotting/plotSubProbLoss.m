function []=plotsubProbLoss(binpc,subProbLab,meanlastp)
% function []=plotsubProbLoss(binpc,subProbLab)
% binpc - [ 2 x L x L ] where 
%      binpc(1,:,i)=perf of i'th classifier on positive labels for all sub-probs
if ( nargin<2 ) error('Insufficient number of arguments'); end;
if ( nargin<3 ) meanlastp=0; end;

if ( ndims(binpc)>3 )
   % use the mean over the sets
   binpc(~isfinite(binpc))=0;% nan-protection   
   binpc_mu=sum(binpc,4)./sum(binpc>0,4);
   binpc_se=sqrt(sum(binpc.^2,4)-binpc_mu.^2)./sum(binpc>0,4);
   binpc   =binpc_mu;
else
   binpc_se=[];
end
holdp=ishold; if ( ~holdp ) cla; hold on; end;
for i=1:size(binpc,3);
   % build the problem description string, if not given
   if( i>numel(subProbLab) )
      subProbLab{i}='mean';
   elseif( iscell(subProbLab{i}) && numel(subProbLab{i})==2 )
      if ( isnumeric(subProbLab{i}{1}) )
         subProbLab{i}{1}=sprintf('%d ',subProbLab{i}{1});
      end
      if ( isnumeric(subProbLab{i}{2}) )
         subProbLab{i}{2}=sprintf('%d ',subProbLab{i}{2});
      end
      subProbLab{i}=[subProbLab{i}{1} ' vs ' subProbLab{i}{2}];
   end
   
   % store so we know to include it in the legend later
   if( isempty(binpc_se) ) 
      h(i)=plot(binpc(1,1:end-meanlastp,i),linecol(i),...
                'LineWidth',3,'DisplayName',[subProbLab{i}]);      
      hold on;      
      if ( meanlastp ) 
         plot(size(binpc,2),binpc(1,end,i),[linecol(i) '.'],...
              'MarkerSize',25,'DisplayName',[subProbLab{i}]);
      end;
      
      if ( size(binpc,1)>1 ) % second row is -class value
         plot(binpc(2,1:end-1,i),[linecol(i) '--'],...
              'LineWidth',1,'DisplayName',[subProbLab{i}]);
         plot(size(binpc,2),binpc(2,end,i),[linecol(i) 'o'],...
              'MarkerSize',7,'DisplayName',[subProbLab{i}]);
      end
   
   else % we've got a error bar so use it!
      h(i)=errorbar(binpc(1,1:end-meanlastp,i),binpc_se(1,1:end-1,i),linecol(i),...
                    'LineWidth',3,'DisplayName',[subProbLab{i}]);
      if ( meanlastp ) 
         hold on;
         errorbar(size(binpc,2),binpc(1,end,i),binpc_se(1,end,i),...
                  [linecol(i) '.'],'MarkerSize',25,'DisplayName',[subProbLab{i}]);
      end
      
      if ( size(binpc,1)>1 ) % second row is -class value
         errorbar(binpc(2,1:end-meanlastp,i),binpc_se(2,1:end-1,i),...
                 [linecol(i) '--'],'LineWidth',1,'DisplayName',[subProbLab{i}]);
         if ( meanlastp )
         errorbar(size(binpc,2),binpc(2,end,i),binpc_se(2,end,i),...
                 [linecol(i) 'o'],'MarkerSize',7,'DisplayName',[subProbLab{i}]);
         end
      end
   end
end;
%axis([0 1 0 .5],'autox');
set(gca,'XTick',1:numel(subProbLab),'XTickLabel',subProbLab);
%xlabel('Test Problem'); ylabel('error rate (%)');
legend(h); legend boxoff
grid on
drawnow
if ( ~holdp ) hold off; end;