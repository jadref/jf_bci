function [shrinkage,sigma]=shrinkDiag(x,dim,shrink)

% function sigma=covdiag(x)
% x (t*n): t iid observations on n random variables
% sigma (n*n): invertible covariance matrix estimator
%
% Shrinks towards diagonal matrix
% if shrink is specified, then this constant is used for shrinkage
if ( nargin<2 ) dim=1; end;
% estimate sample parameters
if ( dim==1 ) x=reshape(x,size(x,1),[])'; dim=2;
elseif ( dim==ndims(x) ) x=reshape(x,[],size(x,ndims(x))); dim=2;
else error('Only 1st/last dim supported');
end

% de-mean returns
[t,n]=size(x);
meanx=mean(x);
x=x-meanx(ones(t,1),:);

% compute sample covariance matrix
sample=(1/t).*(x'*x);

% compute prior
prior=diag(diag(sample));

if (nargin < 3 | shrink == -1) % compute shrinkage parameters
  
  % what we call p 
  y=x.^2;
  phiMat=y'*y/t-2*(x'*x).*sample/t+sample.^2;
  phi=sum(sum(phiMat));  
  
  % what we call r
  rho=sum(diag(phiMat));
  
  % what we call c
  gamma=norm(sample-prior,'fro')^2;

  % compute shrinkage constant
  kappa=(phi-rho)/gamma;
  shrinkage=max(0,min(1,kappa/t));
    
else % use specified constant
  shrinkage=shrink;
end

% compute shrinkage estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;

