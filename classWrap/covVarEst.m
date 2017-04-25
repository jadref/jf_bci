function var=covVarEst(X,dim,varThresh,kernelp)
% estimate the variance/radius of the data as outlier-rejected variance estimate
%
% var=covVarEst(X,dim,varThresh,kernelp)
%
% Inputs:
%  X   - [n-d] data to estimate scaling for, examples in dimension dim
%  dim - [int] dimension(s) which contain examples (ndims(X))
%  varThresh - threshold in std-deviations used for outlier rejection in 
%              variance/radius estimation (4)
%  kernelp   - [bool] flag that input is a kernel
if ( nargin < 2 || isempty(dim) ) dim=ndims(X); end;
if ( nargin < 3 || isempty(varThresh) ) varThresh=4; end;
var=1;
dim=dim(:); % ensure col vector
szX=size(X);
var=0;
if ( all(dim(:)==(1:numel(dim))') ) % leading examples
	error('Not implemented yet!');
elseif ( all(dim(:)==numel(szX)-(numel(dim):-1:1)'+1) ) % trailing examples
  diagIdx=int32(1:size(X,1)+1:size(X,1)*size(X,1));
  X=reshape(X,[size(X,1)*size(X,2) prod(szX(dim))]);
  var=0;for ei=1:size(X,2); var=var+sum(abs(X(diagIdx,ei))); end; 
  var=var./szX(end);
end
