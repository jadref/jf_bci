function [Wb,f,J]=swlda(X,Y,C,varargin)
% stepwise linear discriminant classifier
%
% [wb,f,J,p,M]=swlda(X,Y,C,y2s,varargin)
%
% Inputs:
%  X        -- [n-d] data matrix
%  Y        -- [Nx1] +1/0/-1 class indicators
%  C        -- [1x1] regularisation weight (ignored)
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
%  penter   -- [float] p-value to enter the selected set       ([])
%  premove  -- [float] p-value to remove from selected set     ([])
%  scale    -- [bool] re-scale features to unit-std-dev        (0)
% Outputs:
%  wb       -- [] parameter matrix
%  f        -- [Nx1] set of decision values
%  J        -- [1x1] obj fn value
opts=struct('dim',ndims(X),'verb',1,'maxIter',[],'penter',[],'premove',[],'scale',0);
opts=parseOpts(opts,varargin);

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;

% re-shape X to the size that the program wants
szX=size(X); 
if ( numel(dim)>1 ) error('Cant deal with n-d trials'); end;
if ( dim==1 )             X=reshape(X,szX(1),[]);
elseif ( dim==ndims(X) )  X=reshape(X,[],szX(end))';
else
   X=permute(X,[dim 1:dim-1 dim+1:ndims(X)]);
   X=reshape(X,[],szX(dim));
end
if ( isempty(opts.maxIter) ) opts.maxIter=size(X,2);end; % iteration limit

bi=0;
for spi=1:size(Y,2);
   % remove the excluded points from the training set
   incIdx=Y(:,spi)~=0; 
   if( ~all(incIdx) )   Ytrn=(Y(incIdx,spi));  Xtrn=(X(incIdx,:));
   else                 Ytrn=(Y(:,spi));       Xtrn=(X); 
   end;
   sclstr='off'; if ( opts.scale ) sclstr='on'; end;
   % call stepwisefit to do the training
   if ( ~exist('stepwisefit','builtin') && exist('stepwisefit2','file') ) % use local version only if necessary
      [Wbi,SE,PVAL,in,stats] =...
          stepwisefit2(Xtrn,Ytrn(:),'display','off',...
                       'maxiter',opts.maxIter,'penter',opts.penter,'premove',opts.premove,'scale',sclstr);
   else
      [Wbi,SE,PVAL,in,stats] =...
          stepwisefit(Xtrn,Ytrn(:),'display','off',...
                      'maxiter',opts.maxIter,'penter',opts.penter,'premove',opts.premove,'scale',sclstr);
   end
   % N.B. in Wbi even the non-inclued features (which should have weight=0) have a weighting computed
   %  thus to get the final weighting zero out these features
   Wbi(~in)=0; % remove non-included features
   
   % get the solution
   f(:,spi) = X*Wbi; 
   % optimise b
   if ( 0 ) 
     bi = optbias(Ytrn,f(incIdx,spi));   % min classification error
   else
     bi = mean(Ytrn-f(incIdx,spi));        % min least squares error (what stepwise fit does internally)
   end
   f(:,spi) = f(:,spi)+bi;   % update the solution
   % extract the solution information
   Wb(:,spi)=[Wbi;bi];% linear weighting
end

% compute the predictions for all classes/directions
J = 0;

return;
%-------------------------------------------------------------------------------------
function testCases()
%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

[Wb,f,J]=swlda(X,Y,0);
plotLinDecisFn(X,Y,Wb(1:end-1,:),Wb(end,:));



