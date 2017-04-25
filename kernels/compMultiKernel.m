function K = compMultiKernel(X,Z,kerType,varargin)
%K = compMultiKernel(X,Z,kerType,[options,par1,par2,...])
%
%Inputs:
% X: Input objects: [N1 x d]
% Z: Input objects: [N2 x d]
% par1, par2 ... : additional hyper-parameters to use in the kernel computation
% kerType: kernel type. this is either an actual kernel matrix, or String.
%      'linear'  - Linear           K(i,j) = X(i,:)*Z(j,:)
%      'poly'    - Polynomial       K(i,j) = (X(i,:)*Z(j,:)+par(2))^par(1)
%      'rbf'     - RBF              K(i,j) = exp(-|X(i,:)-Z(j,:)|^2/(2*par))
%      'par'     - par holds kernel K(i,j) = par(i,j)
%      'x'       - X holds kernel   K(i,j) = X(i,j)
%      @kerfn    - function_handle to function 'func' such that : 
%                      K(i,j)=func(X,Z,varargin)
%Options:
% dim -- the dimension along which trials lie in X,Z
% grpDim -- the dimensions along which we wish to sub-set X,Z to compute sub-kernels
%           e.g. use: 'grpDim',3 to compute a kernel for each value along X's 3rd dimension
% grpIdx -- [bool size(X,grpDim) x nK] indices of the members of X to go in each sub-kernel
% 
% N.B. use nlinear, npoly, nrbf, etc. to compute the normalised kernel, 
%      i.e. which has all ones on the diagonal.
%
%   varargin: parameter(s) of the kernel.
%
%Outputs:   
% K: [N x N x nK] the computed set of kernel matrices
%
% Version:  $Id$
% 
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied
opts=struct('dim',1,'grpDim',[],'grpIdx',[]);
[opts,varargin]=parseOpts(opts,varargin);
if (nargin < 1) % check correct number of arguments
   error('Insufficient arguments'); return;
end
if ( nargin< 2 ) Z=[]; end;
if ( nargin< 3 ) kerType='linear'; end;
if ( isinteger(X) ) X=single(X); end % need floats for products!
% empty Z means use X
isgram=false;
if ( isempty(Z) ) Z=X; isgram=true; elseif ( isinteger(Z) ) Z=single(Z); end; 
dim=opts.dim; 
if ( isempty(dim) ) dim=1; end; dim(dim<0)=dim(dim<0)+ndims(X)+1; 

% check if just want a normal kernel
if ( isempty(opts.grpDim) || size(opts.grpIdx,2)==1 && all(opts.grpIdx==1) )
   if ( isgram ) 
      K=compKernel(X,[],kerType,'dim',dim,varargin{:}); 
   else
      K=compKernel(X,Z,kerType,'dim',dim,varargin{:}); 
   end;
   return;
end

grpDim=opts.grpDim; grpDim(grpDim<0)=grpDim(grpDim<0)+ndims(X)+1;
if ( numel(grpDim)>1 ) error('multiple group dims not supported yet!'); end;
grpIdx=opts.grpIdx; if ( isempty(grpIdx) ) grpIdx=eye(size(X,grpDim))>0; end; % kernel per element
nK=size(grpIdx,ndims(grpIdx));
szX=size(X);
K  =zeros([prod(szX(dim)),prod(szX(dim)),nK],class(X));
idx={}; for d=1:ndims(X); idx{d}=1:szX(d); end; % index expr
for ki=1:nK;
   idx{grpDim}=grpIdx(:,ki);
   if ( isgram ) 
      K(:,:,ki)=compKernel(X(idx{:}),[],kerType,'dim',dim,varargin{:});
   else % deal with given 2nd args
      idx2=idx; for d=dim; idx2{d}=1:size(Z,d); end;
      K(:,:,ki)=compKernel(X(idx{:}),Z(idx2{:}),kerType,'dim',dim,varargin{:});
   end
end
return;
%----------------------------------------------
function testCase()
X=randn(10,10,100);
grpIdx=eye(size(X,1))>0; % each element is it's own group
K=compMultiKernel(X,[],'linear','dim',-1,'grpDim',1,'grpIdx',grpIdx);
K2=tprod(X,[3 -1 1],[],[3 -1 2]); mad(K,K2) % compute in 2nd way, and check give same result
grpIdx=randn(10,30)>0; % rand subsets of dim 1
