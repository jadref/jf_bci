function [D]=malDis(X,Z,sigma,dim);
% Compute Maliabonis Distances between sets of points
%
% D = malDis(X,Z[,sigma,dim])
% Inputs:
%  X     - n-d matrix of N points
%  Z     - n-d matrix of M points, if empty then Z=X
%  sigma - [size(X,1:ndims(X)-1) x size(X,1:ndims(X)-1)] scaling matrix
%  dim   - dimensio of X,Z which contains the examples
% Outputs:
%  D     - [ N x M ] matrix of point pairwise distances
if ( nargin < 2 ) Z=X; end;
if ( nargin < 3 ) sigma=[]; end;
if ( nargin < 4 ) dim=1; end;
if ( dim < 0 ) dim = ndims(X)+dim+1; end;

% Compute the appropriate indexing expressions
idx  = [-(1:dim-1) 1 -(dim+1:ndims(X))]; % normal index
tidx = [-(1:dim-1) 2 -(dim+1:ndims(X))]; % transposed index

if ( isempty(sigma) ) % no scaling matrix
   XZ=tprod(X,idx,Z,tidx);
   X2=tprod(X,idx,X,idx);
   if( isempty(Z) ) Z2=X2; else Z2=tprod(Z,idx,[],idx); end

else % with scaling matrix
   Xsigma=tprod(X,[-(1:dim-1) dim -(dim+1:ndims(X))],...
                sigma,[-(1:dim-1) -(dim+1:ndims(X)) 1:dim-1 dim+1:ndims(X)]);
   XZ=tprod(Xsigma,idx,Z,tidx);
   X2=tprod(Xsigma,idx,X,idx);
   if( isempty(Z) ) Z2=X2; else
      Zsigma=tprod(Z,[-(1:dim-1) dim -(dim+1:ndims(X))],...
                  sigma,[-(1:dim-1) -(dim+1:ndims(X)) 1:dim-1 dim+1:ndims(X)]);
      Z2=tprod(Zsigma,idx,Z,idx);
   end
end   

% Finally compute the distance matrix
D=repop(X2,repop(-2*XZ,Z2','+'),'+');
