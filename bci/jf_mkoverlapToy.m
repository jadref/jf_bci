function z=jf_mkoverlapToy(varargin);
% make a toy signal produced as a set of overlapping responses
%
% z=jf_mkoverlapToy(varargin);
%
% Options:
%  nEpoch    -- [1x1] number of epochs
%  nClass    -- [1x1] number of classes
%  nSamp     -- [1x1] number of samples for each epoch
%  nRespSamp -- [1x1] number of samples for the stimulus response
%  s2n       -- signal to noise ratio
%  sigType   -- IRF signal type as parameters to mkSig
%  noiseType -- noise signal as parameters to mkSig
%  y2s       -- label to stimulus sequence
%  isi       -- inter-stimulus interval in samples (1)
%  Y  -- [nEpoch x 1] the labels for the simulated data
% Outputs:
%  X  -- [nSamp x nEpoch] the simulated data
%  Y  -- [nEpoch x 1] the labels for the simulated data
%  M  -- [nSamp x nRespSamp x nClass] the mapping from labels to stimulus types at each time
%  y2s  -- [nSamp x nClass] per class stimilus matrix
%  xtrue -- [nSamp x nClass] the true responses without noise
%  ptrue -- [nRespSamp x 1] the true stimulus response
%
%Examples:
%  
%

% opts = struct('nCh',10,'nEpoch',300,'nClass',2,'fs',128,'nSamp',100,'nResSamp',25,'s2n',.1,...
%               'sigType',{{'prod' {'exp' .2} {'sin' 5}}},...
%               'noiseType',{{'coloredNoise',1}},'plot',0);
opts=struct('fs',128,'plot',0,'isi',1,'y2s',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% add the isi to the stim-sequ if wanted
stimTime_samp=[];
if ( ~isempty(opts.y2s) && opts.isi > 1 ) 
  y2s=zeros(size(opts.y2s,1)*opts.isi,size(opts.y2s,2)); 
  stimTime_samp = ((0:size(opts.y2s,1)-1)*opts.isi)+1; % record stim times
  y2s(stimTime_samp,:) = opts.y2s; % re-sample
else
  y2s=opts.y2s;
  stimTime_samp=1:size(opts.y2s,1);
end

% Make a toy data-set -- call mkOverlapToy to construct the actual data
[X,Y,M,y2s,xtrue,ptrue,overlapOpts]=mkoverlapToy('y2s',y2s,varargin{:});
if ( isempty(opts.y2s) ) % record the y2s used
  opts.y2s=y2s; 
  stimTime_samp=1:size(opts.y2s,1);
end; 

X = single(X);
di= mkDimInfo(size(X),'ch',[],[],'time','ms',[1:size(X,2)]/opts.fs*1000,'epoch');
di(2).info.fs=opts.fs; % rec sample rate
info=struct('M',M,'y2s',y2s,'xtrue',xtrue,'ptrue',ptrue);
summary=sprintf('overlapToy nAmp=[%s]',sprintf('%4.2f ',overlapOpts.nAmp(1:min(end,4))));
z = jf_import('toy/overlap','default','overlaptoy',X,di,'summary','');

% set-up the sub-prob
markerdict(1)=struct('name','A',...
                  'key',-1,...
                  'y2s',y2s(:,1),'code',y2s(:,1));
markerdict(2)=struct('name','B',...
                  'key',1,...
                  'y2s',y2s(:,2),'code',y2s(:,2));
for i=1:size(z.X,n2d(z,'epoch')); % for when varies per epoch
  target = Y(i);
  z.di(n2d(z,'epoch')).extra(i).target      = target;
  z.di(n2d(z,'epoch')).extra(i).stimSeq     = opts.y2s; % [ nStim x nSym ]
  z.di(n2d(z,'epoch')).extra(i).stimTime_ms = (stimTime_samp)'*1000./opts.fs; % [ nStim x 1 ]
end
z.di(n2d(z,'epoch')).info.stimSeq=opts.y2s; % for when fixed for all epochs [nStim x nSym]
z.di(n2d(z,'epoch')).info.stimTime_ms = (stimTime_samp)'*1000./opts.fs; % [nStim x 1]
z=jf_addClassInfo(z,'dim','epoch','Y',Y);
z = jf_addprep(z,mfilename,summary,mergeStruct(opts,overlapOpts),info);
return;
%-----------------------------------------------------------------------------
function testCase()
z=jf_mkoverlapToy()
z=jf_mkoverlapToy('nCh',10)