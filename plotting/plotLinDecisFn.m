function []=plotLinDecisFn(phiX,Y,w,b,limits,varargin)
% plot a linear input decision function and its classifier
% 
%[]=plotLinDecisFn(phiX,Y,w,b,varargin)
% Inputs:
%  phiX -- [N x d] matrix of n-d points
%  Y    -- [N x 1] set of point labels
%  w    -- [d x L] the linear decision function weights
%  b    -- [1 x L] the decision function bias term
%  limits--[2 x d] limits to use for plotting each of the dimensions
%  varargin -- other stuff to feed to the labScatPlot command
%
% SeeAlso: labScatPlot
if ( nargin < 5 ) limits=[]; end;
if ( ~isnumeric(limits) ) varargin={limits varargin{:}}; limits=[]; end;
if ( size(w,2)==size(phiX,1) && size(w,1)==1 ) w=w'; end;
cols=['rbgcmy']';
isheld=ishold;
labScatPlot(phiX,Y,varargin{:});
nplot=floor(min(size(phiX,1)/2,5));
pw=ceil(sqrt(nplot)); ph=ceil(nplot/pw);
if ( nplot ) 
   muX=mean(phiX,2); 
   if ( isempty(limits) ) minX=min(phiX,[],2); maxX=max(phiX,[],2);
   else minX=limits(1,:); maxX=limits(2,:);
   end
   for p=1:nplot;
      if ( nplot>1 ) subplot(pw,ph,p); end; 
      dims=p*2-1:p*2; 
      hold on;
      ndim=setdiff(1:size(phiX,1),dims); 
      if ( size(w,2)==1 )
         rest=muX(ndim)'*w(ndim);
         drawLine(w(dims),b  +rest,minX(dims),maxX(dims),'k','LineWidth',2);
         drawLine(w(dims),b+1+rest,minX(dims),maxX(dims),cols(2),'LineWidth',1)
         drawLine(w(dims),b-1+rest,minX(dims),maxX(dims),cols(1),'LineWidth',3)
      else
         for i=1:size(w,2);
           rest=muX(ndim)'*w(ndim,i);
			  col =cols(mod(i-1,numel(cols))+1,:);
            drawLine(w(dims,i),b(i)+rest,minX(dims),maxX(dims),col,'LineWidth',2);
         end
      end
   end
else % 1-D is special case as x is point ID
   minX=[1]; maxX=[size(phiX,2)];
   hold on;
   drawLine(w,b,  minX,maxX,'k','LineWidth',2);
   drawLine(w,b+1,minX,maxX,cols(2),'LineWidth',1);
   drawLine(w,b-1,minX,maxX,cols(1),'LineWidth',3);
end
if(isheld) hold('on'); else hold('off'); end;
return;
%-----------------------------------------------------------------------------
function []=testCase()
ndim=4;N=400;
Y=sign(randn(N,1));X=randn(ndim,N);w=randn(ndim,1);b=randn(1,1);
plotLinDecisFn(X(1,:),Y,w(1),b);     % 1D
plotLinDecisFn(X(1:2,:),Y,w(1:2),b); % 2D
plotLinDecisFn(X,Y,w,b);             % 4D
