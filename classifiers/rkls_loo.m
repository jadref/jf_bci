function [f_loo]=rkls_loo(f,Y,C,U,S,V,tol)
if ( nargin< 7 || isempty(tol) ) tol=1e-6; end;
sS=S; sS(abs(S)<max(abs(S))*tol)=0; % prevent rounding errors
KiGjj = (U(1:numel(f),:).^2)*(sS./(S+C));
f_loo = (f(:)-KiGjj.*Y)./(1-KiGjj);
% % compute in the *slow* way to validate the above equation is correct
% if ( nargout>1 )
%   for i=1:numel(f);
%     Ytrn=Y; Ytrn(i)=0;
%     [ans,tmp]=rkls(U(1:numel(f),:)*diag(S)*V,Ytrn,C);
%     f_lootrue(i,1)=tmp(i);
%   end
% end
return;

function []=testcase()
% true LOO estimate of the prediction
for i=1:size(K,1);
  Ytrn=Y; Ytrn(i)=0;
  [ans,tmp]=rkls(K,Y,C);
  f_lootrue(i)=tmp(i);
end