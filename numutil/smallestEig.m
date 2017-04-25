function [lambda,x]=smallestEig(A,x,tol,beta)
if ( nargin < 2 || isempty(x) ) x=A(:,1)/norm(A(:,1)); end% initial guess
if ( nargin < 3 || isempty(tol) ) tol=1e-4; end;
if ( nargin < 4 || isempty(beta) ) beta=largestEig(A,x,tol); end;
B=A'*A - A*beta - A'*beta+ eye(size(A))*beta*beta; %=(A-beta*eye(size(A,1)))^2;
[lambda,x]=largestEig(B,x,tol);
tx=A*x;
lambda=norm(tx)/norm(x);
if ( x'*tx < 0 ) lambda=-lambda; end;