function [nWins,h]=plotAlgComp(cr,algnms,shownWins,drawEps,setnm,varargin)
%  function [nWins,h]=plotAlgComp(cr,algnms,shownWins,drawEps,setnm,varargin)
%
%  N.B. if called multiple times will accumulate points and win-counts
%
% Inputs:
%  cr - [N x nAlg] set of classification rates per sub-problem
%  algnms - {nAlg x 1} set of algorithm names
%  shownWins  - [bool] plot the shownWins scores? (0)
%  drawEps - [float] tolerance for deciding it's a draw
%  setnm - [str] name for this set of points (when>1 set points on a plot)
if ( nargin < 3 ) shownWins=0; end;
if ( nargin < 4 || isempty(drawEps) ) drawEps=.0; end;
if ( nargin < 5 ) setnm=''; end;
if ( nargin < 6 ) varargin{1}='.'; end;
if ( max(cr(:))<=1 ) mm=min(min(cr(cr(:)>0)),.5); xlim=[mm 1]; ylim=[mm 1]; 
else xlim=[50 100]; ylim=[50 100]; end;
if ( ndims(cr)>2 ) cr=squeeze(cr); end;
if ( size(cr,1)==numel(algnms) ) cr=cr'; end;
nPlot=(size(cr,2)*(size(cr,2)-1))/2;
w=floor(sqrt(1.5*nPlot)); h=ceil(nPlot/w);
ploti=0;
for i=1:size(cr,2)-1;
   for j=i+1:size(cr,2);
      ploti=ploti+1;
      if ( nPlot>1 ) subplot(h,w,ploti); end;
		hold on; grid on; axis equal; axis([xlim ylim]); plot([0 1*xlim(2)],[0 1*ylim(2)]); 
      validPts=all(cr(:,[i j])>0,2);
      h=plot(cr(validPts,i),cr(validPts,j),varargin{:}); 
		if ( ~isempty(setnm) ) set(h,'DisplayName',setnm); end;

      nWins=[sum(cr(validPts,i)>cr(validPts,j)+drawEps); sum(cr(validPts,i)+drawEps<cr(validPts,j))];
      %nWins=nWins+.5*sum(cr(validPts,i)<=cr(validPts,j)+drawEps & cr(validPts,i)+drawEps>=cr(validPts,j));
      
      % ud=get(gca,'userdata'); if(~iscell(ud))ud={ud}; end;
      % if(~isempty(ud) && isequal(ud{1},'nWins') ) 
      %   onWins=ud{2}; ud(1:2)=[]; 
      %   nWins = nWins + onWins(:,min(end,ploti)); 
      % end;
      % set(gca,'userdata',{'nWins' nWins ud{:}});

      if ( ~isequal(shownWins,false) ) % only put summary info in when wanted
        % compute significance 
        if ( isnumeric(shownWins) && size(shownWins,1)== 2 ) nWins=nWins + shownWins(:,min(end,ploti)); end;
        if ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.01) )     sig='**'; % .01
        elseif ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.05) ) sig='*';  % .05
        else sig=''; end; 
        winhdls=findobj(gca,'type','text');
        if ( isempty(winhdls) ) 
          text(.99*xlim(2),.51*ylim(2),[num2str(nWins(1)) sig],'HorizontalAlignment','right','VerticalAlignment','bottom','FontSize',14,'FontWeight','bold');
          text(.51*xlim(2),.99*ylim(2),[num2str(nWins(2)) sig],'HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold');
        else
          set(winhdls(2),'string',[num2str(nWins(1)) sig]); %N.B. hdls are in last-first order!
          set(winhdls(1),'string',[num2str(nWins(2)) sig]);                    
        end
        xlabel(algnms{i},'interpreter', 'none');ylabel(algnms{j},'interpreter','none');
      end
   end
end
