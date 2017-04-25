function [A]=tucker(S,varargin);
U=varargin;
X2=S; nd=numel(U);
for d=1:nd; X2=tprod(X2,[1:d-1 -d d+1:nd],U{d}(:,1:size(X2,d)),[d -d],'n'); end
