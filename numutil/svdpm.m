function [varargout]=svdpm(X,rank,tol,maxIter,verb,seed);
% singular value decomposition - using the power method
%
% [U,S,V]=svdpm(X,rank,tol,maxIter,verb,seedU)
%
% Compute singular value decomposition approximation to the input
% matrix X using the power method 
% (4-6x faster than svds, 100sx faster than svd for v.large matrices)
%
% Inputs:
%  X - [w x h] matrix to decompose
%  rank - [int] number of output components wanted
%  tol  - [2x1] convergence tolerance (1e-2)
%  maxIter - [int] max number of power method iterations (100)
%  verb    - [int] verbosity level
%  seedU   - left hand decomposition to start from ([])
% Outputs:
%  U,S,V   - rank component svd decomposition of X

if ( nargin<2 || isempty(rank) )   rank=1; end;
if ( nargin<3 || isempty(tol) )    tol=1e-2; end;
if ( nargin<4 || isempty(maxIter)) maxIter=100; else maxIter=min(maxIter,1e4); end;
if ( nargin<5 || isempty(verb) ) verb=0; end;
if ( nargin<6 ) seed=[]; end;
if ( numel(tol)<2 ) tol(2)=1e-8; end;
if ( numel(tol)<3 ) tol(3)=1e-10; end;
U=[];S=[];V=[]; SV=1;
for ri=1:rank;
  if ( ~isempty(seed) ) u= seed(:,min(end,ri));
  else                  u= mean(X(:,1:min(end,100)),2); % seed is quick average of 1st few columns
  end
  for iter=1:maxIter; % power method to find left and right singular vectors
    ou=u;
    v=u'*X;
    % deflate the other components, i.e. ortho w.r.t. bits already found
    if( ri>1 ) v=v-u'*U*SV'; end; 
    u=X*v';
    % deflate the other components, i.e. ortho w.r.t. bits already found
    if( ri>1 ) u=u-U*(SV'*v'); end;
    nu=u'*u;
    u=u./sqrt(max(nu,eps));
    % convergence test
    du = sum(abs(ou-u)); % change in vector
    if ( verb>0 ) fprintf('%d.%d) nu=%g du=%g\n',ri,iter,nu,du); end;
    if ( iter<=2 ) du0=du; elseif ( du<tol(1)*du0 ) break; end; % relative tol test
    if ( du<tol(min(2,end)) || nu<tol(min(3,end)) ) break; end; % abs tol test
  end
  v=(u'*X)';
  if( ri>1 ) v=v-(u'*U*SV')'; end; % deflate the other components
  s=sqrt(v'*v);
  v=v./s;
  % upate the sets
  U=[U u];
  V=[V v];
  S=[S;s];
  SV=repop(S','*',V);
end
if ( nargout<=1 ) 
  varargout={S}; 
else
  varargout={U,S,V};
end
return;
%------------------------------------------
function testCase();
X=randn(10,8);
[U,S,V]=rank1sv(X,1);
[U,S,V]=rank1sv(X,2);