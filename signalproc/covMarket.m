function [shrinkage,sigma]=covMarket(x,dim,shrink)

% function [shrinkage,sigma]=covmarket(x)
% x (t*n): t iid observations on n random variables
% sigma (n*n): invertible covariance matrix estimator
%
% This estimator is a weighted average of the sample
% covariance matrix and a "prior" or "shrinkage target".
% Here, the prior is given by a one-factor model.
% The factor is equal to the cross-sectional average
% of all the random variables.

% The notation follows Ledoit and Wolf (2003)
% This version: 06/2009
if ( nargin<2 ) dim=1; end;
% estimate sample parameters
if ( dim==1 ) x=reshape(x,size(x,1),[])'; dim=2;
elseif ( dim==ndims(x) ) x=reshape(x,[],size(x,ndims(x))); dim=2;
else error('Only 1st/last dim supported');
end

% de-mean returns
t=size(x,1);
n=size(x,2);
meanx=mean(x);
x=x-meanx(ones(t,1),:);
xmkt=mean(x')';

sample=cov([x xmkt])*(t-1)/t;
covmkt=sample(1:n,n+1);
varmkt=sample(n+1,n+1);
sample(:,n+1)=[];
sample(n+1,:)=[];
prior=covmkt*covmkt'./varmkt;
prior(logical(eye(n)))=diag(sample);

if (nargin < 3 | shrink == -1) % compute shrinkage parameters
  c=norm(sample-prior,'fro')^2;
  y=x.^2;
  p=1/t*sum(sum(y'*y))-sum(sum(sample.^2));
  % ris divided into diagonal
  % and off-diagonal terms, and the off-diagonal term
  % is itself divided into smaller terms 
  rdiag=1/t*sum(sum(y.^2))-sum(diag(sample).^2);
  z=x.*xmkt(:,ones(1,n));
  v1=1/t*y'*z-covmkt(:,ones(1,n)).*sample;
  roff1=sum(sum(v1.*covmkt(:,ones(1,n))'))/varmkt...
	  -sum(diag(v1).*covmkt)/varmkt;
  v3=1/t*z'*z-varmkt*sample;
  roff3=sum(sum(v3.*(covmkt*covmkt')))/varmkt^2 ...
	  -sum(diag(v3).*covmkt.^2)/varmkt^2;
  roff=2*roff1-roff3;
  r=rdiag+roff;
  % compute shrinkage constant
  k=(p-r)/c;
  shrinkage=max(0,min(1,k/t));
else % use specified number
  shrinkage = shrink;
end

% compute the estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;



