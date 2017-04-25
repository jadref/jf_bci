function [ndx]=subv2ind(sz,subs);
cp=[1 cumprod(sz(1:end-1))]';
ndx=(subs-1)*cp+1;