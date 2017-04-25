function [type,bgns,ends,offset_samp,opts]=timedPhases(status,varargin)
% Identify the trail start/ends based on timing
%
% [type,bgns,ends,offset_samp]=comprecphases(status,...)
%
% Inputs:
%  PresentationPhase -- [Nx1] vector of presentation phase values at each sample
% Options:
%  fs         -- [int] sample rate of the data
%  start_ms/start_samp  -- [nWin x 1] start index/time for the windows
%  trlen_ms/trlen_samp  -- [1 x 1] number of samples/ms to put in the window
%  offset_ms/offset_samp-- [2x1 int] offset time in ms before/after to include in trial
%  nwindows
%  overlap
% Outputs:
%  type -- [M x 1] id of the epoch type
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
opts=struct('nwindows',[],'overlap',[],'start_ms',[],'start_samp',[],'trlen_ms',[],'trlen_samp',[],'fs',[],...
            'offset_ms',[],'offset_samp',[]);
opts=parseOpts(opts,varargin);

% deal with other options
if ( ~isempty(opts.fs) ) 
   samp2ms = 1000/opts.fs; 
   ms2samp = opts.fs/1000;
end

% Use the given trial length to over-ride the status info if wanted
if ( ~isempty(opts.start_ms) )
   if ( isempty(opts.fs) ) error('no fs: cant compute ms2samp'); end;
   opts.start_samp = floor(opts.start_ms*ms2samp);
end
if ( ~isempty(opts.trlen_ms) )
   if ( isempty(opts.fs) ) error('no fs: cant compute ms2samp'); end;
   opts.trlen_samp = floor(opts.trlen_ms*ms2samp);
end
opts.trlen_samp=min(opts.trlen_samp,numel(status));% limit to datafile size

% do the epoch computation
[bgns,width]=compWinLoc(numel(status),opts.nwindows,opts.overlap,opts.start_samp,opts.trlen_samp);
ends=bgns+width;

% offset if wanted
if ( ~isempty(opts.offset_ms) ) 
   if ( numel(opts.offset_ms)<2 ) 
      opts.offset_ms=[-opts.offset_ms opts.offset_ms]; 
   end;
   opts.offset_samp = ceil(opts.offset_ms*ms2samp);
end
if ( ~isempty(opts.offset_samp) && ~isequal(opts.offset_ms,0) )
   bgns=min(max(bgns+opts.offset_samp(1),1),numel(status));
   ends=min(max(ends+opts.offset_samp(2),1),numel(status));   
   offset_samp=opts.offset_samp(1); % sample idx of bgn relative to 0 in the epoch
else 
   offset_samp=0;
end

type=ones(size(bgns));
return;

%-------------------------------------------------------------------------------------
function testCases()
[bg,en]=timedPhases(zeros(100,1),'nwindows',10);
[bg,en]=compRecPhases(zeros(100,1),5,-5)
