function [K,wbs]=linMapKer(K,V,lambda,wbs);
% Re-scale kernel, and dual-spec points along spec directions
%  K' = K + V*diag(lambda-1)*V'  
%
% [K,wbs]=linMapKer(K,V,lambda,wbs);
%
% Inputs:
%  K -- [N x N] kernel matrix
%  V -- [N x M] set of directions to map
%  lambda -- [M+1 x 1] set of relative new sizes for these directions.  M+1'th value is for non-spec dir
%            N.B. must be positive and >0
%  wbs -- [N x d] set of directions which should be inverse transformed such that:
%                 K*wbs = Kdef*wbsdef
% Outputs
%  Kdef   -- [N x N] transformed kernel
%  wbsdef -- [N x d] the inverse transformed pts
if( nargin < 4) wbs=[]; wbsdef=[]; end;
if( numel(lambda)==1 ) K=lambda*K; return; end; % deal with spherical special case
lambda0=lambda(end); lambda=lambda(1:end-1); lambda=lambda(:);
Kv  = K*V;  % project kernel on the deflation directions
vKv = tprod(Kv,[-1,2],V,[-1 2],'n'); % length in feature space of these dirs
V   = repop(V,'./',sqrt(vKv));   % make unit length -- in feature space
Kv  = repop(Kv,'./',sqrt(vKv)); % make unit length -- in feature space
K   = lambda0*K + repop(Kv,'*',lambda')*Kv'; % apply the transformation

if ( ~isempty(wbs) ) % inverse transform the wbs 
   wbsKv = wbs'*Kv; % project each pt onto deflation dir
   wbs=(wbs - V*diag(lambda./(lambda0+lambda))*wbsKv')./lambda0;
end
return;
%------------------------------------------------------------------------------------