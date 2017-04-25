function [Wm,U,D]=maxPowDir(W,Cxx,dim)
% find the max-power projection of C onto the set of spatial-filters W
% 
% [W,dir]=maxPowDir(W,Cxx,dim)
%
% Inputs:
%  W    -- [nCh x nOri x nFilt] set of spatial filters
%  Cxx  -- [nCh x nCh] covariance of the data
%  dim  -- [int] dims of W to use, dim(1)=chDim, dim(2:end)=maxPowDim(s)
% Outputs:
%  W    -- [nCh x 1 x nFilt] set of spatial filters with nOri removed
%  U    -- [nOri x nFilt] orientation used for each filter
%  D    -- [nFilt] power in this direction
if (nargin < 3 || isempty(dim)) dim=[1 2]; end;
szW=size(W);
tmp= szW; tmp(dim(2:end))=1; Wm = zeros(tmp,class(W));
U  = zeros([prod(szW(dim(2:end))) szW(setdiff(1:end,dim))]);
for fi=1:prod(szW(setdiff(1:end,dim)));
   WCxxW = W(:,:,fi)'*Cxx*W(:,:,fi);
   [Ufi,Dfi]=eig(WCxxW); Dfi=diag(Dfi); [Dfi,si]=sort(Dfi); Ufi=Ufi(:,si); % get the max-var dir
   U(:,fi) = Ufi(:,1); D(fi)=Dfi(1);   % max variance/power direction
   Wm(:,1,fi)  = W(:,:,fi)*Ufi(:,1);     % filter for the max var/power dir
end
return;
%----------------------------------------------------------------------------------
function testCase()
