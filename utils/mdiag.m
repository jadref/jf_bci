function [diagX]=mdiag(X,dims)
% multi-dimensional generalisation of diag -- element extraction
%
% function [diagX]=mdiag(X,dims)
if ( nargin < 2 || isempty(dims) ) dims=1:2; end;
dims(end+1:2)=dims(1)+1;
dims=sort(dims,'ascend');

if( diff(dims)~=1 ) error(['Only for consequetive dims currently sorry!']); end
   
% reshape X to make the indexing easy.
sz=size(X);
X =reshape(X,[prod(sz(1:dims(1)-1)) prod(sz(dims)) prod(sz(dims(2)+1:end))]);
diagIdx=1:sz(dims(1))+1:min(sz(dims)).^2;
diagX = X(:,diagIdx,:);
diagX = reshape(diagX,[sz(1:dims(1)-1) sz(dims(1)) 1 sz(dims(2)+1:end)]);
return;
%----------------------------------------------------------
   