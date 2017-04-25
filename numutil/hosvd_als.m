function [S,varargout]=hosvd_als(A,r,maxIter,tol,varargin)
% Compute a rank-n tucker-decomposition of an n-d array
%
% [S,U1,U2,...UN] = hosvd_als(A,[rank,maxIter,tol,S,U1,U2,...UN])
% Compute the rank limited Tucker decomposition of the input n-d matrix
% using the optimising least squares routine to (attempt to) make this the
% best rank r1..rN approx)
% This provides a decomposition of an n-d matrix A as:
%
%  A_{i1,i2,...iN} = S^{j1,j2,...jN} U1_{i1,j1} U2_{i2,j2} ... UN_{iN,jN}
% where, S=[r1 x r2 x ... x rN] array and Uj=[size(A,j) x rj]
%
% Where we use Einstein Summation Convention to indicate implicit
% summation over the repeated labels on the RHS.
%
% For more information see:
% See: "Tutorial on MATLAB for tensors and the Tucker
%  decomposition", T. G. Kolda, B. Bader, Workshop on Tensor
%  Decomposition and Applications
%  
% Inputs:
%  A  -- a n-d matrix
%  rank -- [1x1] or [n x 1] vector of ranks for the individual component
%          decompositions. (1)
%  maxIter -- maximum number of iterations to perform
%  tol-- tolerance on change in U, to stop iteration
%  S  -- [1x1] seed higher-order eigen-value
%  U1 -- [ size(A,1) x 1 ] starting value for A's 1st dimension
%  U2 -- [ size(A,2) x 1 ] starting value for A's 2nd dimension
%  ...
%  UN -- [ size(A,N) x 1 ] starting value for A's Nth dimension
% Outputs:
%  S  -- an n-d "all-orthogonal" matrix of "singular values"
%  U1 -- a [size(A,1) x size(A,1)] matrix of orthogonal vectors
%  U2 -- a [size(A,2) x size(A,2)] matrix of orthogonal vectors
%  ...
%  UN -- a [size(A,N) x size(A,N)] matrix of orthogonal vectors

sizeA=size(A); nd=ndims(A);
if ( nargin < 2 || isempty(r) ) r=1; end;
if ( nargin < 3 || isempty(maxIter) ) maxIter=10; end
if ( nargin < 4 || isempty(tol) ) tol=1e-3; end;

if ( numel(r) < nd ) r(end+1:nd)=1; end;
% Fill in the (rest of) the seed values
if ( isempty(S) ) S=1; end;
if ( numel(U)<nd )
   [idx{1:nd}]=deal(1); % build index expr to extract the bit we want from A
   for d=numel(U)+1:nd;      
      idx{d}=1:sizeA(d);
      U{d}=shiftdim(A(idx{:}));  U{d}=U{d}./norm(U{d}); % seed guess
      if ( r(d) > 1 ) 
         U{d}=repmat(U{d},1,r(d));
         % Add noise for symetry breaking
         U{d}=U{d} + randn(size(U{d}))*norm(U{d}(:,1))*1e-3;  
      end
      idx{d}=1;
   end
end

% Loop over A's dimensions computing in individual SVD's
for iter=1:maxIter;
   
   AA = A; % Temp store of the accumulated product up to this dim
   deltaU=0;
   for d=1:nd;
      oU=U{d};
      
      tA=AA;
      for d2=d+1:nd; % Cached A^{1:N} U_1..U_{d-1} in AA so only mult the rest
         tA=tprod(tA,[1:d2-1 -d2 d2+1:nd],U{d2},[-d2 d2]);
      end
      
      if ( r(d)==1 ) % Normal power method
         U{d}=shiftdim(tA);
         U{d}=U{d}/norm(U{d});  % Normalise the final vector
      
      else  % Power method using svd decomposition to factorise stuff
         % See: Tut on MATLAB for tensors, p 35
         U{d}=svds(reshape(permute(tA,[d,1:d-1,d+1:ndims(tA)]),size(tA,d),prod(size(tA))/size(tA,i)),r(d));
      
      end

      deltaU = deltaU+sum(abs(oU(:)-U{d}(:))); % info on rate of change of bits

      AA = tprod(AA,[1:d-1 -d d+1:nd],U{d},[-d d]); % update the accumulated info
   end   

   S  = tprod(tA,[1:nd-1 -nd],U{nd},[-nd],'n'); % N.B. assume about tA

   if ( 1 ) % verb > 0 ) 

      A2 = S;
      for d=1:nd;
         A2 = tprod(A2,[1:d-1 -d d+1:nd],U{d},[d -d],'n');
      end
      
      mad=max(abs(A(:)-A2(:))); sse=sum((A(:)-A2(:)).^2);
      fprintf('%2d)\tdeltaU=%8g\tmad=%8g\tsse=%8g\n',iter,deltaU,mad,sse);
   end

   if ( deltaU < tol ) break; end;
   
end

% Finally compute the multiplier
S = tprod(tA,[1:nd-1 -nd],U{nd},[-nd nd]);
varargout=U;
return;

%------------------------------------------------------------------------------
function testCase()
X = randn(10,10,10);
[S,U{1:ndims(X)}]=hosvd_als(X); Ssvd=S; Usvd=U;

% Visualise the solution
image3d(S); figure; mimage(U{:});

% Check the decomposition works
X2=S; 
for d=1:numel(U); X2=tprod(X2,[1:d-1 -d d+1:numel(U)],U{d}(:,size(X2,d)),[d -d],'n'); end
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
   figure(r); image3d(Xlr{r});
end
