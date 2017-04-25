function [X]=mxcat(X,d,t)
% remove the matrix dimension d by conconating it entries along dimension
% t
% $Id: mxcat.m,v 1.5 2006-11-16 18:23:47 jdrf Exp $
if ( nargin < 2 ) d=2; end; % defaults
if ( nargin < 3 ) t=1; end;
if ( t==d ) return; end;
sizeX=size(X);ndim=ndims(X); 
if ( t<d ) pred=[1:t-1];         postd=[t+1:d-1 d+1:ndim];
else       pred=[1:d-1 d+1:t-1]; postd=[t+1:ndim];
end
X=permute(X,[pred,t,d,postd]);                              % d to after t
X=reshape(X,[sizeX(pred) sizeX(t)*sizeX(d) sizeX(postd)]);  % cat the dims

return
%-----------------------------------------------------------
function []=testCases()
X=ones(2,3,4,5,6);
size(mxcat(X,1))
size(mxcat(X,2))
size(mxcat(X,3))
size(mxcat(X,4))
size(mxcat(X,5))
size(mxcat(X,3,2))
size(mxcat(X,3,4))
size(mxcat(X,3,5))


