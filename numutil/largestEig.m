function [lambda,x]=largestEig(A,x,tol)
if ( nargin < 2 || isempty(x) ) x=A(:,1)/norm(A(:,1)); end% initial guess
if ( nargin < 3 || isempty(tol) ) tol=1e-2; end;
xold = x + 1;                  % for convergence test
lambda=1; olambda=inf;
while abs(lambda-olambda) > tol*lambda; % until eigvalue convergence
  % min(norm(x-xold),norm(x+xold)) > tol ,  % until eigvector convergence
  xold = x;                    % keep old one
  tx = A*x;                    % map through A
  olambda=lambda; lambda= tx'*x;  % get approx to eigenvalue
  x = tx/norm(tx);             % normalise again
                               %fprintf('%g\n',norm(tx)/norm(xold));
end
lambda= (A*x)'*x;  % compute final eigenvalue
