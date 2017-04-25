function [chol,err]=cholFactorise(A)
% Compute the lower triangular cholesky factorisation of A.
if ( norm(A-A') > 1e-9 ) error('Only symetric pos-def matrices accepted');end
chol=zeros(size(A));
for i=1:size(A);
   [chol(i,1:i) err]=cholLUpdate(chol(1:i-1,1:i-1),A(1:i,i));
   if (err) return; end;
end
err=[];