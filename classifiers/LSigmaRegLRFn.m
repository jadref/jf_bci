function [J,df,ddf,obj,dv,S,U]=LSigmaLRFn(wb,X,Y,C)
% the objective function for the LSigma Reg loistic-regression function
szX=size(X);
W=reshape(wb(1:end-1),szX(1:end-1));
b=wb(end);
if ( ndims(W)==2 )
   [U{1},S,U{2}]= svd(W,'econ'); S=diag(S);
else
   [S,U{1:ndims(W)}]=parfac_als(W,'rank',20,'verb',0,'C',1e-5);
end
dv   = tprod(X,[-(1:ndims(X)-1) 1],W,-(1:ndims(W))) + b;
g    = 1./(1+exp(-(Y.*dv))); g=max(g,eps);

Ed   = -sum(log(g));  % -ln(P(D|w,b,fp))
Ew   = sum(abs(S));   % L_sigma
J    = Ed + C(1)*Ew;
df   = [];
ddf  = [];
obj  = [J Ed Ew];
return;
