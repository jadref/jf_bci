function [y2s]=timedStimuli(stimSeq,stimTimes,sampTimes)
% convert set of stimuli types and times at which these stimuli occur to per-sample indicators/marker ch.
%
% Inputs:
%  stimSeq   - [nEvents x nStim x nSeq] set of stimulus events
%  stimTimes - [nEvents x nSeq] per-epoch times at which these stimulus events happened
%  sampTimes - [nSamp x nSeq] times at which each sample was recorded for each epoch (min(stimTimes):max(stimTimes))
% N.B. in all cases the nSeq dimension values are replicated if it is of size 1
% Outputs:
%  y2s       - [nSamp x nStim x nSeq] per-sample event indicator
if ( nargin<3 || isempty(sampTimes) ) sampTimes=[min(stimTimes(:)):max(stimTimes(:))]'; end;
if ( max(size(stimTimes))==numel(stimTimes) && size(stimTimes,1)~=size(stimSeq,1) ) stimTimes=stimTimes(:); end;
if ( max(size(sampTimes))==numel(sampTimes) && size(sampTimes,2)~=size(stimSeq,3) ) sampTimes=sampTimes(:); end;
%if ( ndims(stimTimes)<3 ) stimTimes=reshape(stimTimes,[size(stimTimes,1) 1 size(stimTimes,2)]); end;
% combine these 2 to get the per-symbol/sample stimulus matrix
y2s = zeros([size(sampTimes,1) size(stimSeq,2) size(stimSeq,3)]);%[nSamp x nStim x nSeq]
idx=[];
for si=1:size(stimSeq,3);
  % get sample idx of these times
  if ( isempty(idx) || ~all(stimTimes(:,min(end,si))==stimTimes(:,min(end,si-1))) ) % cache index computation
    for i=1:size(stimTimes,1)
      [ans,ii]=min(abs(sampTimes(:,min(end,si))-stimTimes(i,min(end,si))));
      idx(i)=ii;
    end
  end
  % plus markers only. plus/minus markers
  y2s(idx,:,si)=stimSeq(:,:,si);      % stim events
%  y2s(idx,si,:,2)=-(stimSeq(:,:,si)-1); % non-stim events
end
