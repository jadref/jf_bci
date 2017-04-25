function []=labScatPlot(X,Y,w,varargin)
% function []=labScatPlot(X,Y,w,...)
% X -- [dim x N ] pts
% Y -- [N x 1] pts labels
%      OR
%      [L x N] pts indicator
% varargin -- additional options to pass to the plot command
%
% $Id: labScatPlot.m,v 1.17 2007-07-03 23:35:42 jdrf Exp $
maxsubplots=5;
if ( numel(findobj(gcf, 'type', 'axes')) > 1 ) maxsubplots=1; end;
if ( nargin < 3 ) w=[]; end;
if ( ~isnumeric(w) ) varargin={w varargin{:}}; w=[]; end;
zeroLab=false;
if ( numel(varargin)>0 && isequal(varargin{1},'zeroLab') ) zeroLab=true; varargin=varargin(2:end); end;
holdp=ishold();
cols=['rbgcmy']';
symbs=['o+x*sdv^<>ph.']';
if ( nargin < 2 || isempty(Y) ) % normal scatter plot..
  if ( size(X,2) > 1 || ~isreal(X) ) 
     if ( ~isreal(X) ) Xp=[real(X(1,:));imag(X(1,:))]; else Xp=X([1 2],:); end;
    plot(Xp(1,:),Xp(2,:),symbs(1),'Color',cols(1));
    if(~isempty(w)) plot(Xp(1,abs(w)>0),Xp(1,abs(w)>0),'o','Color',cols(1),varargin{:}); end
  else
    plot(X(1,:),symbs(1),'Color',cols(1),varargin{:});
  end
else
  if ( size(Y,2)==size(X,2) ) Y=Y'; end;
  if ( min(size(Y)) == 1 )
     uY=unique(Y); 
     if ( ~zeroLab ) uY(uY==0)=[]; end; % zero lab ignored
     if ( numel(uY) < 16) 
        Y=lab2ind(Y,uY,[],1,0); 
     else     
        % cut the data up into grey-scale sets and plot these        
        cols=colormap();%gray(50);        
        if( numel(uY)<size(cols,1) )
           cols=cols(round(linspace(1,size(cols,1),numel(uY))),:);
        end
        % TODO: should really use histogram equalized bins...
        binsY=linspace(min(Y),max(Y),min(numel(uY),size(cols,1))); %cut Y into equ col bins
        for i=1:numel(binsY)-1;
           labY(:,i)= single(Y > binsY(i) & Y <= binsY(i+1));
        end;
        Y=labY;
     end;
  end
  exInd=all(Y==0,2); Y=Y(~exInd,:); X=X(:,~exInd); % remove excluded points
  nY=size(Y,2); indsp=false;
  if ( 0 && ~(all(sum(Y>0,2)==1) && all(sum(Y<0,2)==size(Y,2)-1)) ) % indep sub-probs
     nY=size(Y,2)*2; indsp=true;
  end
  for li=1:nY;
     if ( indsp ) l=ceil(li/2); else l=li; end;
    if ( ~isempty(w)) mw=max(abs(w)); end;
    if ( any(Y(:,l)~=0) )
      if ( indsp && mod(li,2) ) 
         pts=logical(Y(:,l)<0);
      else         
         pts=logical(Y(:,l)>0);
      end
      col=cols(mod(l-1,length(cols))+1,:);
      symb=symbs(mod(l-1,length(symbs))+1,:);
      if ( size(X,1) > 1 || ~isreal(X) ) 
         nplot=min(floor(size(X,1)/2)*(1+~isreal(X)),...
                   maxsubplots); % up to 10d plots
         pw=ceil(sqrt(nplot)); ph=ceil(nplot/pw);
         for p=1:nplot; % do each consequetive pair of dims
            if ( nplot > 1 ) subplot(pw,ph,p); end;           
            if ( ~isreal(X) ) % extract the bit we're putting in ths plot
               Xp=[real(X(p,:));imag(X(p,:))];
            else
               d=(p-1)*2+1;
               Xp=[X(d,:);X(d+1,:)];
            end               
            plot(Xp(1,pts),Xp(2,pts),symb,'Color',col,varargin{:});hold on;
            if ( ~isempty(w) ) 
               wpts=find(pts & abs(w(:)) > 0);
               maxx=max(X,[],2);minx=min(X,[],2);maxr=min(maxx-minx)/4;
               for wpt=wpts'
                  r=abs(maxr*w(wpt)/mw);
                  rectangle('Position',[Xp(:,wpt)'-r/2  r r],'Curvature',[1 1]);
               end               
               plot(Xp(1,pts & abs(w(:))>0),Xp(2,pts & abs(w(:))>0),'o','Color',col); 
            end
         end
      else
        plot(find(pts),X(1,pts),symb,'Color',col,varargin{:}); hold on ;
      end
    end
  end
end
if ( ~holdp ) hold('off'); end; 
return;
%-------------------------------------------------------------------------
% TESTCASES:
function testCase()
% 1) Labelled:
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 20],[.3 .3; .3 .3; .2 .2]);
labScatPlot(X,Y);
% 3) complex
labScatPlot(complex(X(1,:),X(2,:)),Y)
% 2) Grey
dim=3;f=[-.5 .5 1]'; X=(eye(dim,dim)+f*f')*randn(dim,1000);
labScatPlot(X(1:2,:),X(3,:));
