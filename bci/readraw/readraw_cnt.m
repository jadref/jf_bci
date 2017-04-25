function [x,di,fs,summary,opts,info] = readraw_bdf(filename, varargin)
% Function to read a raw-bdf file, slice it and return the data and meta-info
%
% [x,di,fs,summary,opts,info] = readraw_bdf(filename, varargin)
% Inputs:
%  filename       -- the bdf file to read
% Options:
%  epochFn        -- [str] name of function to use to compute epoch start/ends ('compRecPhases')
%                     [type,bgns,ends,offset]=epochFn(markerSequence,RecPhaseLimits{:})
%  RecPhaseLimits -- {2x1 cell} first number is phase start and second is phase end
%  offset_ms      -- [2x1 int] offset time in ms before/after to include in trial
%  trlen_ms       -- [int] Force trials to be this long (from start marker)
%  prune_equal    -- [bool] prune all trials to be the same length
%  single         -- [bool] store data in single format
%  chanIdx        -- [nCh x 1 bool] or [int] Indices of the channels to keep
%  subsample      -- [struct] sub-sample per trial
%  statusZero_samp -- [int] number of samples to ensure status sets to 0 before new markers
% Outputs
%  X              -- [numel(chanIdx)+1 x nSamp x nEpoch] data matrix
%  di             -- [ndims(X)+1 x 1] dimInfo struct describing the layout of X
%  fs             -- [float] sampling rate of this data
%  summary        -- [str] string describing what we loaded
%  opts           -- [struct] the parsed options structure
%  info           -- [struct] containing additional useful info about th eloaded data
opts = struct('prune_equal',1,'epochFn','compRecPhases',...
              'single',1,'verb',1,'chanIdx',[],'subsample',struct('fs',256),...
              'statusZero_samp',2);
[opts,varargin]=parseOpts(opts,varargin);

x = [];
y = [];
fs = [];
di = [];
summary = '';
info=struct('filename',filename); % record file name

hdr=sopen(filename); % get the header info
info.hdr = hdr;

if any(hdr.SampleRate~=hdr.SampleRate(1))
   error('channels with different sampling rate not supported');
end
fs  = hdr.SampleRate(1);
samp2ms = 1000/fs; 
ms2samp = fs/1000;

% downsampling?
subSampRatio=1;
if ( ~isempty(opts.subsample) && opts.subsample.fs < fs ) 
   subSampRatio = fs/opts.subsample.fs;
end

nchannels = hdr.NS;
nRec      = hdr.NRec;
sampPerRec= hdr.SPR;
nsamp     = nRec * sampPerRec;

% extract the event info
events=hdr.EVENT;

% convert to a virtual status channel
status = zeros(nsamp,1,'int16');
evPos=events.POS; evType=events.TYP;
for ei=1:length(evPos)-1;   
   status(evPos(ei):evPos(ei+1)-1)=evType(ei);
   if( evType(ei)==evType(ei+1) ) %N.B. -2 so we leave the transition to 0 in place
      status(max(evPos(ei)+1,evPos(ei+1)-1-opts.statusZero_samp*subSampRatio):evPos(ei+1)-1)=0;
   end
end
status(evPos(end)+1:end)=status(evPos(end)); % last marker to eof is same phase

% Now we can extract the bits we want
[epType,bgns,ends,offset_samp]=feval(opts.epochFn,status,varargin{:},'fs',fs);

if ( isempty(bgns) || isempty(ends) ) 
   warning('File: %s didnt contain any valid epochs',filename);
   return;
end
trlens=ends-bgns+1;
% BODGE: check for badly finished file with over-long last trial
longTr=find(trlens>10*median(trlens));
if ( ~isempty(longTr) )
   warning(sprintf('%d **Very** long trial(s) found, may be CORRUPT. Deleted',numel(longTr)));
   epType(longTr)=[];bgns(longTr)=[];ends(longTr)=[];
end

maxtrlen=max(trlens(:));
if ( opts.prune_equal )
   maxtrlen=min(trlens(:));
   ends  =bgns+maxtrlen-1;           % set end to min length
end

%---------------------------------------------------------------------------
% Finally: Extract the bits we want
%---------------------------------------------------------------------------
ntrials = numel(bgns);
fprintf('Identified %d epochs to read.\n',ntrials);
chanIdx = opts.chanIdx; if ( isempty(chanIdx) ) chanIdx=1:nchannels; end;
if ( islogical(chanIdx) ) chanIdx=find(chanIdx); end;
%if ( ~any(chanIdx==statusch) ) chanIdx(end+1)=statusch; end;
if ( isfield(hdr,'Chan_Select') )
   hdr.Chan_Select(:)=0; hdr.Chan_Select(chanIdx)=1;
end
nchannels=numel(chanIdx);
nchannels=nchannels+1; % extra channel for the status inf
y = double(epType);

% downsampling?
if ( subSampRatio>1 )
   info.ofs=fs; 
   fs=opts.subsample.fs; info.fs=fs;
   fprintf('Downsampling: from %d to %d.\n',info.ofs,info.fs);
   omaxtrlen = maxtrlen;       maxtrlen   = floor(maxtrlen./subSampRatio);
   ooffset_samp=offset_samp;   offset_samp= floor(offset_samp./subSampRatio);
end

if ( opts.single ) x = zeros(nchannels, maxtrlen, ntrials, 'single');
else               x = zeros(nchannels, maxtrlen, ntrials);
end
for tr=1:ntrials;
   bgntr= bgns(tr);
   endtr= min(max(bgns(tr)+maxtrlen,ends(tr)),nsamp); % use all data available, i.e. don't 0-pad
   xtr  = sread(hdr,(endtr-bgntr)*samp2ms/1000,bgntr*samp2ms/1000)'; % N.B. time x ch returned!
   str  = single(status(bgntr:endtr)); % N.B. use fixed status channel
   if ( subSampRatio>1 ) % sub-sample
      [xtr,idx] = subsample(xtr,size(xtr,2)./subSampRatio,2);
      str = str(min(end,max(1,round(idx))));
   end
   x(1:size(xtr,1),1:size(xtr,2),tr)  =xtr;
   x(end,1:size(xtr,2),tr)=str;
   if ( opts.verb > 0 ) % progress reporting
      si=round(tr/(ntrials/100)); % nearest step
      if ( tr==round(si*ntrials/100) ) 
         fprintf(repmat('.',max(1,si-round((tr-1)./(ntrials/100))),1)); 
      end
   end;
end

% Now setup the dim info array
if ( isfield(hdr,'Label') ) % use names for the channels if available 
   chvals=cellstr(hdr.Label);
   if ( ~isempty(chanIdx) ) chvals=chvals(chanIdx); end;
else
   chvals=opts.chanIdx;
end
% include status channel
if ( iscell(chvals) ) chvals={chvals{:} 'Status'}; end;
di = mkDimInfo(size(x),'ch','',chvals,...
               'time','ms',([1:maxtrlen]+offset_samp(1))*samp2ms*subSampRatio,...
               'epoch','',[],'','uV');
% restrict the states to the start of the trials too & record as trialinfo
for tr=1:ntrials;
   di(3).extra(tr).sampleIdx = [bgns(tr) ends(tr) -offset_samp(min(end,tr),:)];
   di(3).extra(tr).marker    = y(tr);
end
