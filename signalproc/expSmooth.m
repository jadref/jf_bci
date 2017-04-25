function mu=expSmooth(mu,smoothDim,smoothFactor)
% expionentially smooth the input
%
% N.B. half life = ln(.5)./ln(smoothFactor)
%      smoothFactor = exp(ln(.5)./halfLife)
if ( isempty(smoothFactor) || smoothDim==0 ) return; end;
idx={}; for d=1:ndims(mu); idx{d}=1:size(mu,d); end;     
idx{smoothDim}=1;
musi = mu(idx{:});
for si=2:size(mu,smoothDim);
  idx{smoothDim}=si;
  mu(idx{:}) = (smoothFactor)*musi + (1-smoothFactor)*mu(idx{:});
end
return;
