function [X,di,fs,summary,opts,info] = readraw_bcpy2000(filename, varargin)
% read a BCI2000 data file and slice it into epochs
%
% A modified version of readraw_bci2000 to cope with the different state
% recording conditions used for the bcpy2000 format
%
% N.B. Default PresentationPhase value mapping is:
%      0   - inter-trial gap                (always)
%      1   - recording phase                (always)
%      2,3 - timing traffic lights
%      4   - cue
%      5   - idle phase
%
%
% [x,di,fs,summary,opts,info] = readraw_bcpy2000(filename, varargin)
% Inputs:
%  filename -- the bci2000 file to read
% Options:
%  TargetState    -- [str] the name of the state vector which contains target labels
%  PresentaionPhase--[str] the state vector state name which contains the current
%                    presentation phase info.  Used to slice the data
%  RecPhaseLimits -- first number is phase start and second is phase end
%  ignoreStates   -- [cell of str] list of states don't record info about
%  perTrialStates -- [cell of str] list of state names for which only per-epoch info is recorded
%  synccorrect    -- [bool] flag if we should try to correct transition times using VEStimulusInfo
%  TimeStampState -- [str] name of the state which contains stim correction time info
%  rescale        -- [bool] rescale the data with stored gain/offset info
%  offset_ms      -- offset time in ms before/after to include in trial
%  trlen_ms       -- Force trials to be this long (from start marker)
%  prune_equal    -- prune diff length trials to be the same length
%  single         -- store data in single format
%  chanIdx        -- Indices of the channels to keep
%  subsample      -- sub-sample per trial
% Outputs
%  X              -- [numel(chanIdx)+1 x nSamp x nEpoch] data matrix
%  di             -- [ndims(X)+1 x 1] dimInfo struct describing the layout of X
%  fs             -- [float] sampling rate of this data
%  summary        -- [str] string describing what we loaded
%  opts           -- the parsed options structure
%  info           -- [struct] containing additional useful info about th eloaded data

opts =struct('TargetState','StimulusCode','PresentationPhase','StimulusCode','RecPhaseLimits',1,...
    'IgnoreStates',{{'AppStartTime','Recording','Running'}},...
    'perTrialStates',{{'SourceTime' 'StimulusTime' 'VEStimulusTime' 'targetclass' 'CurrentTrial' 'CurrentBlock'}},...
    'synccorrect',1,'TimeStampState','VEStimulusTime',...
    'nonTimeSyncStates',{{'SourceTime' 'StimilusTime' 'VEStimilusTime'}},...
    'rescale',1,'offset_ms',0,'trlen_ms',[],'prune_equal',1,'single',1,'verb',0,'subsample',0);
opts=parseOpts(opts,varargin);

if ( numel(opts.RecPhaseLimits)==1 ) % if not given end phase==begin phase
   opts.RecPhaseLimits{2}=-opts.RecPhaseLimits{1}; 
end
if ( ~iscell(opts.RecPhaseLimits) ) % ensure is cell array
   opts.RecPhaseLimits = {opts.RecPhaseLimits -opts.RecPhaseLimits};
end


X = [];
y = [];
fs = [];
trialinfo = [];
summary = '';
info = [];

%----------------------------------------------------------------------------
% extract the basic info from the file
%----------------------------------------------------------------------------
if ~strncmp(lower(fliplr(filename)), 'tad.', 4), return, end
s    = readbci2000(filename);
info = bci2000param(s); % extra info contains all the params used
fs   = bci2000param(s, 'SamplingRate');
if ( isstr(fs) ) fs=str2num(fs); end;
samp2ms = 1000/fs; 
ms2samp = fs/1000;
pktsz=bci2000param(s, 'SampleBlockSize'); % packet siz
if ( isstr(pktsz) ) pktsz=str2num(pktsz); end;
npackets= floor(size(s.signal,2)/pktsz);
[nch,nsamp]=size(s.signal);

% get the state information
ss = bci2000state(s);
fns=fieldnames(ss);
for i=1:numel(fns); % remove states we ignore
   if ( ~isempty(strmatch(fns{i},opts.IgnoreStates)) ) 
      ss=rmfield(ss,fns{i});
   end; 
end

%----------------------------------------------------------------------------
% fix the timing information using VEStimilusTime
%----------------------------------------------------------------------------
% N.B. sourceTime is time at the START of the current packet!
% a) wraparround of the times
% b) finding the non-first entry start time of the believable VEStimulusTime's
if ( opts.synccorrect & isfield(ss,opts.TimeStampState) )
   timeStamp =double(getfield(ss,opts.TimeStampState));
   sourceTime=double(getfield(ss,'SourceTime'));
   % find first index with valid contents, i.e. 1 packet after running pressed
   % which means first non-zero timeStamp after zero time-stamps
   timeStampZeroIdx=pktsz+find(timeStamp(pktsz+1:end)~=0,1,'first');
   % make 1st entry 0 time & first valid entry time of that sample
   if(double(ss.SourceTime(timeStampZeroIdx))<timeStamp(timeStampZeroIdx)||...
      double(ss.SourceTime(timeStampZeroIdx-1)>timeStamp(timeStampZeroIdx)) )
      % jump to the next transition...
      timeStampZeroIdx=timeStampZeroIdx+find(timeStamp(timeStampZeroIdx+1:end)~=timeStamp(timeStampZeroIdx),1,'first');
   end
   % zero out everything before this time, its garbage anyway
   timeStamp (1:timeStampZeroIdx-1)=0;
   sourceTime(1:timeStampZeroIdx-1)=0;   
   
   % Now we've got the transitions as sample indices we can convert the states
   % to be correct times at which transition happened   
   fns=fieldnames(ss); % remove states we shouldn't sync-correct
   i=1;
   while i < numel(fns); 
      if ( ~isempty(strmatch(fns{i},opts.nonTimeSyncStates)) ) fns(i)=[]; end; 
      i=i+1;
   end;
   
   dfsIdx=find(diff([0 timeStamp])~=0);   % find the state changes
   for i=1:numel(dfsIdx);
      pcktIdx  = dfsIdx(i);               % index of the start of this packet
      if ( sourceTime(pcktIdx) < timeStamp(pcktIdx) ) 
         warning('Stimilus time info @%d samp is corrupt',pcktIdx); continue;
      end;      
      % Correct for the difference by copying values from now back in time to
      % when the transition actually happened
      % difference between sourceTime and VEStimilusTime tells us how far back
      % to move the transition
      transIdx=pcktIdx-round((sourceTime(pcktIdx)-timeStamp(pcktIdx))*ms2samp);
      srcIdx  =pcktIdx;
      % The following is a test for a bug in BCPy2000 which meant that it 
      % included a state transition in the PACKET before it should have done.
      % If we detect this then instead of copying back the new value we copy
      % forward the old value.
      if( ss.(opts.PresentationPhase)(transIdx) == ss.(opts.PresentationPhase)(pcktIdx) )
         warning('Corrected state transition timing bug @%d samp',pcktIdx);
         tmp=transIdx; transIdx=pcktIdx-pktsz; pcktIdx=tmp; 
         srcIdx=transIdx-1; % copy old correct state values forward
      end
      for i=1:numel(fns)
         ss.(fns{i})(transIdx:pcktIdx)=ss.(fns{i})(srcIdx);
      end
   end
   
end

%----------------------------------------------------------------------------
% Identify the trail start/ends
%----------------------------------------------------------------------------
% Now extract the bit we need from the read data file
% We extract the bit where PresentationPhase matches the indicated values.
% find the samples which match the appropriate values.
if ( ~isfield(ss,opts.PresentationPhase) )
   error(['PresentationPhase: ' opts.PresentationPhase ' not found']);
end
if ( ~isfield(ss,opts.TargetState) ) 
   error(['TargetState: ' opts.TargetClass 'not found']);
end
PresentationPhase=getfield(ss,opts.PresentationPhase);

% Now we can extract the bits we want
bgns=[]; ends=[];
for j=1:2:numel(opts.RecPhaseLimits);
   [tbgns tends]=compRecPhases(PresentationPhase,opts.RecPhaseLimits{j},opts.RecPhaseLimits{j+1});
   ends(bgns<2*pktsz)=[]; bgns(bgns<2*pktsz)=[];   % ignore the first packet
   bgns=[bgns  tbgns];
   ends=[ends  tends];
end
% sort by bgn point
[bgns,si]=sort(bgns,'ascend'); ends=ends(si); 
trbgns = bgns; % just so we know for sure the status value we started from

% Use the given trial length to over-ride the status info if wanted
if ( ~isempty(opts.trlen_ms) ) 
   ends=min(bgns+floor(opts.trlen_ms*ms2samp),nsamp);
end

% offset if wanted
offset_samp=0;
if ( ~isempty(opts.offset_ms) )
   offset_samp = opts.offset_ms(1)*ms2samp;
   if ( numel(opts.offset_ms)<2 ) 
      opts.offset_ms=[-opts.offset_ms opts.offset_ms]; 
   end;
   bgns=min(max(bgns+ceil(opts.offset_ms(1)*ms2samp),1),nsamp);
   ends=min(max(ends+ceil(opts.offset_ms(2)*ms2samp),1),nsamp);
end

trlens=ends-bgns+1;
maxtrlen=max(trlens(:));
if ( opts.prune_equal )
   maxtrlen=min(trlens(:));
   ends  =bgns+maxtrlen-1;           % set end to min length
end

% Correct the center/scale with the stored info
offset=0; gain=1;
if ( opts.rescale )
   if ( isfield(info,'SourceChGain') ) 
      offset=info.SourceChOffset;
      if ( isstr(offset) ) offset=str2num(offset); end;
      if( all(offset==offset(1)) ) offset=offset(1); end
      offset=offset(:);
      gain  =info.SourceChGain;   
      if ( isstr(gain) ) gain=str2num(gain); end;
      if ( all(gain==gain(1)) )    gain  =gain(1);   end
      gain  =gain(:);
   else
      warning('Cant rescale no SourceChGain parameter found!');
   end
end

%---------------------------------------------------------------------------
% Finally: Extract the bits we want
%---------------------------------------------------------------------------
ntrials = numel(bgns);
nchannels = size(s.signal, 1);
y = getfield(ss,opts.TargetState);
y = double(y(trbgns))';
if ( opts.single ) X = zeros(nchannels, maxtrlen, ntrials, 'single');
else               X = zeros(nchannels, maxtrlen, ntrials );
end
for tr=1:ntrials;
   xtr = double(s.signal(:,bgns(tr):ends(tr)));
   if ( opts.rescale ) % rescale
      xtr = repop(repop(xtr,'-',offset),'.*',gain); 
   end
   X(:,1:size(xtr,2),tr)=xtr;
   if ( opts.verb > 0 ) % progress reporting
      si=round(tr/(ntrials/100)); % nearest step
      if ( tr==round(si*ntrials/100) ) 
         fprintf(repmat('.',si-round((tr-1)./(ntrials/100)),1)); 
      end
   end;
end
s.signal=[]; % clear the old data store

%---------------
% now setup the dimInfo array
chvals=info.ChannelNames;
if ( isempty(chvals) || isscalar(chvals) ) chvals=1:size(X,1); end;
di = mkDimInfo(size(X),'ch','',chvals,...
               'time','ms',([1:maxtrlen]+offset_samp)*samp2ms,...
               'epoch','',[],'','mV');

% restrict the states to the start of the trials too & record as trialinfo
fns=fieldnames(ss);
trialinfo=struct();
for tr=1:ntrials;
   trialinfo(tr).sampleIdx = [bgns(tr) ends(tr) offset_samp];
   for i=1:numel(fns);
      fvals=getfield(ss,fns{i});
      if ( ~isempty(strmatch(fns{i},'CurrentBlock')) )
         trialinfo(tr).block=fvals(trbgns(tr));
      elseif ( ~isempty(strmatch(fns{i},'CurrentTrial')) ) 
         trialinfo(tr).trial=fvals(trbgns(tr));         
      elseif ( ~isempty(strmatch(fns{i},opts.perTrialStates)) )
         trialinfo(tr).(fns{i})=fvals(trbgns(tr));
      elseif ( ~isempty(strmatch(fns{i},opts.TargetState)) )
         trialinfo(tr).marker = fvals(trbgns(tr));
      elseif ( ~isempty(strmatch(fns{i},opts.PresentationPhase) ) )
         % ensure always is a trialinfo var called PresentationPhase
         trialinfo(tr).PresentationPhase = fvals(bgns(tr):ends(tr));
         trailinfo(tr).(fns{i})=fvals(bgns(tr):ends(tr));
      else % record per-sample info
         trialinfo(tr).(fns{i})=fvals(bgns(tr):ends(tr));
      end
   end
end
di(3).extra=trialinfo;

% for summary only rec which states we match
us=unique(PresentationPhase);
summary = [opts.PresentationPhase '= ' ivec2str(intersect(opts.RecPhaseLimits{1},us)) '->' ivec2str(intersect(opts.RecPhaseLimits{2},us))];

return;

function [str]=ivec2str(chr,vec); % pretty print vector of ints
if ( nargin < 2 ) vec=chr; chr=' '; end;
str='';
if(numel(vec)>1) 
   str=sprintf(['%d' chr],vec(1:end-1));
   str=['[' str sprintf('%d',vec(end)) ']'];
else
   str=sprintf('%d',vec);
end;

%-----------------------------------------------------------------------------
function testCase()
% for the pilot read the raw bdf files.
global bciroot; bciroot ={'~/data/bci' '/Volumes/BCI_Data'};

expt       = 'own_experiments/test_experiments/comparative_test';
subjects   = {'rs'};
sessions   = {'20081111' '' '' '' '' '' ''};
dtype      = 'raw_bci2000';

blocks = {1:30 1:30 1:30 1:30 1:30 1:30 1:30 1:30}
markers= {1:100};
blockprefix='';

subj=subjects{1}; session=sessions{1}; block=blocks{1}; marker=markers{1};
filelst = findFiles_mmm(expt,subj,session,block,blockprefix,dtype,'.*R[0-9][0-9]*.dat');


[x,di,fs,summary,opts,info] = readraw_bcpy2000(filelst(1).fname,'RecPhaseLimits',marker)

