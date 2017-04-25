function [S]=featstats(X,dim,wght);
% multi-dimensional variance computation
%
% [s]=featstats(X,dim[,wght]);
% Inputs:
%  X   -- n-d matrix
%  dim -- dimension of X to sum along to compute the variance
%  wght-- [size(X,dim) x L] set of weightings for the points in dim
% Outputs:
%  s   == [size(X) x 2] with (size(s,dim)=L, and mean+var in last dim
if ( nargin < 3 ) wght=[]; end;
sz=size(X);
if ( isempty(wght) || numel(wght)==1 )
   sW  = size(X,dim);
   sX  = sum(X,dim)./sW;
   sX2 = tprod(X,[1:dim-1 -dim dim+1:ndims(X)],[],[1:dim-1 -dim dim+1:ndims(X)])./sW;
else
   wght= repop(wght,'./',sum(wght)); % normalise weights
   sX = tprod(X,[1:dim-1 -dim dim+1:ndims(X)],wght,[-dim dim]); % means
   sX2= tprod(X,[1:dim-1 -dim dim+1:ndims(X)],...
              repop(reshape(X,[sz(1:dim) 1 sz(dim+1:end)]),'.*',shiftdim(wght,-dim+1)),...
              [1:dim-1 -dim dim:ndims(X)]);
end
sX2= sX2 - sX.^2; % var = ( \sum_i x_i^2 - (\sum_i x_i)^2/N ) / N
S  = cat(ndims(X)+1,sX,sX2);
return;
%-------------------------------------------------------------------------
function testCase()
X = randn(100,2); wght=rand(100,3);
featStats(X,1),mean(X),var(X,1)
featStats(X,1,wght(:,1)),X'*wght(:,1)./sum(wght(:,1)),var(X,wght(:,1))