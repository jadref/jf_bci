function [X]=mcov(X,dim,stepDim)
% multi-dimensional covariance computation
%
% [X]=mcov(X,dim,stepDim,type)
%
% Inputs:
%  X    -- [n-d float] the data
%  dim  -- [int] dimension to outer-product over
%  stepDim -- [int] dimensions to step along
%  type -- 'str' type of covariance to compute, n-d or matrix-cov
if( nargin<2 || isempty(dim) )     dim=1; end;
if( nargin<3 || isempty(stepDim) ) stepDim=[]; end;
if( nargin<4 || isempty(type) )    type=''; end;
szX=size(X); nd=ndims(X);
dim=sort(dim,'ascend'); % ensure in ascending order

% Map to co-variances, i.e. outer-product over the channel dimension(s)
% insert extra dim for OP and squeeze out accum dims
shifts=zeros(ndims(X),1); shifts(dim)=2; shifts(stepDim)=1; % change in output dim
idx2  =cumsum(shifts);  idx2(shifts==0)=-(1:sum(shifts==0)); 
idx1  =idx2;            idx1(dim)=idx1(dim)-1; 

oX=X;
if ( isreal(X) || isreal(Y) ) % include the complex part if necessary
   X = tprod(real(X),idx1,[],idx2);
elseif ( ~isreal(X) && ~isreal(Y) )
   X = tprod(real(X),idx1,[],idx2) + tprod(imag(X),idx1,[],idx2);
end

%if ( numel(dim)>1 ) X=X/prod(sz(setdiff(1:end,dim))); end
return
%---------------------------------------------------------------------------------------------
function testCase()
X=randn(10,100,10);
X=cumsum(cumsum(X,1),2); % add some dependency info  
Cxx=mcov(X,[1 2],3);
