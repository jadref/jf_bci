function [M,y2s,xtrue,ptrue,opts]=mkConvSig(varargin)
% make a toy signal produced as a set of overlapping responses
% Options:
%  nClass    -- [1x1] number of classes
%  nSamp     -- [1x1] number of samples for each epoch
%  irflen    -- [1x1] number of samples for the stimulus response
%  y2s       -- [nSamp x nClass] per class stimilus matrix
%  sigType   -- IRF signal type as parameters to mkSig
%  unitnorm  -- [bool] normalise the output signal strength?  (1)
%  preStimVal -- [1x1] default stimulus value for all time before 0 (0)
% Outputs:
%  M  -- [nSamp x nRespSamp x nClass] the mapping from labels to stimulus types at each time
%  y2s  -- [nSamp x nClass] per class stimilus matrix
%  xtrue -- [nSamp x nClass] the convolved response
%  ptrue -- [nRespSamp x 1] the un-convolved true stimulus response
opts = struct('nClass',2,'nSamp',100,'irflen',32,'y2s',[],...
              'sigType',{{'prod' {'exp' .2} {'sin' 5}}},'unitnorm',1,'preStimVal',0);
opts=parseOpts(opts,varargin);

% Build a toy data-set
if ( isempty(opts.y2s) )
   y2s = randn(opts.nSamp,opts.nClass)>.5; y2s(1,:)=0;% [nSamp x nClass] codes with different structures
else
   y2s = opts.y2s; opts.nClass=size(y2s,2); % over-ride num classes
end

% trim y2s to the size of the output
y2s = y2s(1:min(end,opts.nSamp),:,:);
% Pad y2s with enough extra samples for the final sets of responses
y2s = cat(1,y2s,zeros(opts.nSamp-size(y2s,1),size(y2s,2)));

% Now build class specific convolution matrix
M=mkConvMx(y2s,opts.irflen,opts.preStimVal);
% True response
ptrue = mkSig(opts.irflen,opts.sigType{:}); % make the true response signal
% And hence the true response
xtrue = tprod(single(M),[1 -1 2],single(ptrue),[-1],'n'); % true response per class

if ( opts.unitnorm ) % normalise the signal strength
  xtrue = repop(xtrue,'./',max(sqrt(sum(xtrue.*xtrue,1)),eps));
end

return;
%--------------------------------------------------------------------------
function testCase()
[M,y2s,xtrue,ptrue,opts]=mkConvSig();