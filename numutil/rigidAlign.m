function [R, t] = rigidAlign(X, Y)
% compute Least-squares rigid-body trans from X into Y
%
% [R, t] = rigidAlign(X, Y)
%
% Y \approx  R*X + t
%
% Inputs:
%  X -- [d x N] set of points (starting set)
%  Y -- [d x N] set of points (ending set)
% Outputs:
%  R -- [d x d] rotation+scaling matrix
%  t -- [d x 1] translation matrix
muX = mean(X,2);
muY = mean(Y,2);
Cxy = X*Y'-size(X,2)*muX*muY'; % cov XY
[U S V] = svd(Cxy); S=diag(S); % use svd a robust way of computing inverse covariance
% BODGE: should be R=V*U';
R = V*V';           % inv covariance, N.B. assuming Cxy is symetric!
l = real(eig(R));
if ( any(l<0) ) R = V*diag(sign(l))*U'; end;
if any(S<0), error('Illspec inputs'); end;
c = sum(abs(S))/(sum(sum(X.*X))-size(X,2)*muX'*muX); % allow shrinkage only
R = c*R;
t = muY - R*muX;
return
%--------------------------------------------------
X=randn(3,4); 
b=randn(size(X,1),1);
Y=repop(X,'+',b);
[R, t] = rigidAlign(X, Y);
