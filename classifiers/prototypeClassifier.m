function [wb,f,J]=prototypeClassifier(X,Y,C,varargin)
% class centeriod based prototype classifier
%
% [wb,f,J,p,M]=prototypeClass(X,Y,C,varargin)
%
% Inputs:
%  X        -- [n-d] data matrix
%  Y        -- [Nx1] +1/0/-1 class indicators
%  C        -- [1x1] regularisation weight (ignored)
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
% Outputs:
%  wb       -- [] parameter matrix
%  f        -- [Nx1] set of decision values
%  J        -- [1x1] obj fn value
opts=struct('dim',ndims(X),'wght',[1 -1],'verb',1,'alphab',[]);
opts=parseOpts(opts,varargin);

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;

% compute the class centriods
szY=size(Y); if ( szY(end)==1 ) szY(end)=[]; end;
wght=cat(numel(szY)+1,single(Y>0),single(Y<0));
wght=repop(wght,'./',2*shiftdim([(sum(Y(:)>0));(sum(Y(:)<0))],-ndims(wght)+1)); % mean
% merge into single weighting matrix
wght=tprod(wght,[1:numel(szY) -ndims(wght)],opts.wght(:),[-ndims(wght)],'n');

% compute the result
Xidx=1:ndims(X); Xidx(dim)=-dim;
W   = tprod(X,Xidx,wght,-dim,'n');

% compute the predictions on the whole data set
Xidx=-(1:ndims(X)); Xidx(dim)=1:numel(dim);
Widx=-(1:ndims(W)); Widx(dim)=0;
f   = tprod(X,Xidx,W,Widx,'n');

% compute the optimal bias
trnInd=Y~=0;
b = optbias(Y(trnInd),f(trnInd));

% apply the bias & generate return values
f   = f + b;
J   = 0;
wb  = [W(:);b];
return;
%--------------------------------------------------------------------------
function testCase()
%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

[wb,f,J]=prototypeClassifier(X,Y,[]);

% only weight the positive examples
[wb,f,J]=prototypeClassifier(X,Y,[],'wght',[1 0]);

plotLinDecisFn(X,Y,wb(1:end-1),wb(end));

