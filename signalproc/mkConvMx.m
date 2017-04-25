function [M]=mkConvMx(y2s,nResSamp,preStimVal)
% convert per-sample indicator matrix into equivalent convolution matrix
%
%  [M]=mkConvMx(y2s,irflen,preStimVal)
% 
% Inputs:
%  y2s - [nSamp x ....] matrix of per-sample event indicators
%  irflen - [1x1] number of samples to allow for the impluse response
%  preStimVal - [1x1] default value for all time points before stimulus began
% Output:
%  M   - [nSamp x irflen x ... ] convolution matrix
if ( nargin<3 ) preStimVal=[]; end;
szy2s=size(y2s);
M=zeros([szy2s(1),nResSamp,szy2s(2:end)],class(y2s)); % maps from time points to response points & inverse
y2sidx = repop((1:size(y2s,1))','-',(0:nResSamp-1)); % sel sub-sets of C
if ( ~isempty(preStimVal) ) % pad before with pre-stimulus value single non-stimulus event
  y2s=cat(1,repmat(preStimVal,[1 szy2s(2:end)]),y2s); % add extra pre-stim value
  y2sidx=y2sidx;  % update the indicies
end
y2sidx=max(y2sidx,1); % prevent negative/illegal indicies
for yi=1:prod(szy2s(2:end));
   M(:,:,yi) = reshape(y2s(y2sidx,yi),size(M,1),size(M,2)); % [t x tau x ...]
end
return;