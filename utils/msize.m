function [sz]=msize(X,dims)
% extension of size to allow for size along multiple dims
if ( nargin<2 ) dims=1:ndims(X); end
sz=[size(X) 1]; % dims past end of X all have size 1
sz=sz(min(dims,numel(sz)));
return;