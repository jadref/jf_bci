function [X]=imag2cart(Z,dim)
% convert from complex form to cartesian form
%
% [X]=imag2cart(Z,dim)
%
% Inputs
%  Z -- complex input to convert
%  dim -- dimension of output to cat the re,im along (1)
if ( nargin<2 ) dim=1; end
if ( dim<0 ) dim=dim+ndims(Z)+1; end
X=cat(dim,real(Z),imag(Z));
return;