function [S,varargout]=hosvd(A,r)
% Compute the Higher Order SVD decomposition of an n-d array
%
% [S,U1,U2,...UN] = hosvd(A,rank)
% Compute the Higher-Order Singular Value Decompostion (also called the
% Tucker Factorisation) of the input n-d matrix A.
% This provided a decomposition of an n-d matrix A as:
%
%  A_{i1,i2,...iN} = S^{j1,j2,...jN} U1_{i1,j1} U2_{i2,j2} ... UN_{iN,jN}
%
% Where we use Einstein Summation Convention to indicate implicit
% summation over the repeated labels on the RHS.
%
% For a "low-rank", r, approx to A use:
%  A_{i1,i2,..iN} =
%    S^{j1(1:min(r,end),j2(1:min(r,end),...jN(1:min(r,end))}
%       U1_{i1,j1(1:min(r,end))} U2_{i2,j2(1:min(r,end))} ... UN_{iN,jN(1:min(r,end))}
%
% For more information see:
%  L. De Lathauwer, B. De Moor, and J. Vandewalle, "A Multilinear
%  Singular Value Decomposition", SIAM J. Matrix Analysis Applications,
%  Vol 21. No 4. pp. 1253-1278 
%
%  
% Inputs:
%  A  -- a n-d matrix
%  rank -- the rank of the approximation to compute
% Outputs:
%  S  -- an n-d "all-orthogonal" matrix of "singular values"
%  U1 -- a [size(A,1) x size(A,1)] matrix of orthogonal vectors
%  U2 -- a [size(A,2) x size(A,2)] matrix of orthogonal vectors
%  ...
%  UN -- a [size(A,N) x size(A,N)] matrix of orthogonal vectors

sizeA=size(A); nd=ndims(A);
if ( nargin < 2 ) r=[]; end;

% Loop over A's dimensions computing in individual SVD's
S = A; % Used to hold the singular values
A = double(A);
for i=1:ndims(A);
   if ( isempty(r) ) 
      [U{i},D,V] = svd(reshape(permute(A,[i,1:i-1,i+1:ndims(A)]),sizeA(i),prod(sizeA)/sizeA(i)),'econ');
   else  % given max rank
      [U{i},D,V] = svds(reshape(permute(A,[i,1:i-1,i+1:ndims(A)]),sizeA(i),prod(sizeA)/sizeA(i)),r);
   end
   S = tprod(S,[1:i-1 -i i+1:ndims(A)],U{i},[-i i]);
end
varargout=U;
return;

%------------------------------------------------------------------------------
function testCase()
X = randn(10,10,10);
[S,U{1:ndims(X)}]=hosvd(X); Ssvd=S; Usvd=U;

% Visualise the solution
image3d(S); figure; mimage(U{:});

% Check the decomposition works
X2=S; nd=numel(U);
for d=1:nd; X2=tprod(X2,[1:d-1 -d d+1:nd],U{d}(:,1:size(X2,d)),[d -d],'n'); end
max(abs(X(:)-X2(:)))

image3d(X);figure;image3d(X2);

% Check the quality of the decomposition as we reduce the number of
% components we use.
figure(100);image3d(X);
for r=1:max(size(X));
   X2=S(1:min(r,end),1:min(r,end),1:min(r,end)); % The low ranks we want
   for d=1:numel(U); 
      X2=tprod(X2,[1:d-1 -d d+1:numel(U)],U{d}(:,1:size(X2,d)),[d -d],'n');
   end
   Xlr{r}=X2;
   max(abs(X(:)-Xlr{r}(:))),
   %figure(r); image3d(Xlr{r});
end
