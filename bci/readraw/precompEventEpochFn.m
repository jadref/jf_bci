function [epochs,bgns,ends,offset_samp,opts,bgnsi,endsi]=precompEventEpochFn(events,varargin)
% slice the data with the given epoch information, N.B. ignores events information!
%
% [epochs,bgns,ends,offset_samp,opts,bgnsi,endsi]=precompEventEpochFn(events,varargin)
%
% Inputs:
%  events - [struct] vector of events to search.  Events should have the 
%           fields: 
%               'type','value','sample','offset','duration'
%            OR
%               'TYP',  'VAL', 'POS',  'DUR'
%
% Options:
%  epochs -- [struct] vector of epochs events.  These should have the fields:
%               'type','value','sample','offset','duration'          
%  fs         -- [int] sample rate of the data
%  offset_ms  -- [2x1 int] offset time in ms before/after to include in trial
%  trlen_ms   -- [int] Force trials to be this long (from start marker)
% Outputs:
%  epochs -- [M x 1 struct] event which contained the start of epoch info
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
%  bgnsi-- [M x 1] indicies in to the events structure of the start of the epochs
%----------------------------------------------------------------------------
% Identify the trail start/ends
%----------------------------------------------------------------------------
opts=struct('epochs',[],'offset_ms',[],'offset_samp',[],'trlen_ms',[],'trlen_samp',[],'fs',[]);
if ( numel(varargin)>0 && isstr(varargin{1}) && strcmp('RecPhaseLimits',varargin{1})) varargin(1)=[]; end;
[opts,varargin]=parseOpts(opts,varargin);
if ( numel(varargin)>0 ) opts.epochs=varargin{1}; end;
epochs=opts.epochs;

bgns=[]; ends=[]; offset_samp=0; 
bgnsi=1:numel(epochs); endsi=1:numel(epochs);
if ( isempty(epochs) ) return; end;

if ( isfield(epochs,'POS') )
  bgns=[epochs.POS];
  ends=bgns + [epochs.DUR];
  lastSample = max(epochs.POS) + epochs(end).DUR;
elseif ( isfield(epochs,'sample') )
  bgns=[epochs.sample];
  ends=bgns + max(1,[epochs.duration]);
  lastSample = epochs(end).sample+epochs(end).duration;
end
if ( ~isempty(opts.trlen_samp) )  
  ends=bgns+1;
end

% make sure all the bgns are before the ends!
if ( numel(bgns)~=numel(ends) ) 
   warning('Unequal numbers of start and end transitions: %d~=%d',numel(bgns),numel(ends));
   if ( numel(bgns)>numel(ends) ) 
      % more bgns, may be an incomplete last trial, complete it if so
      if ( bgns(numel(ends)+1)>ends(end) )         
         bgns(numel(ends)+2:end)=[]; epochs(numel(ends)+2:end)=[];
         ends(end+1)=nsamp;
         warning(' FIXED last partial trial -- MAY BE CORRUPT!');
      else % can we do anything else?
         error(' Cant fix!');
      end
   elseif ( numel(ends) > numel(bgns) ) 
      % more ends, may be an incomplete first trial incomplete, complete if so
      if ( bgns(1)>ends(1) )
         ends=ends(2:end);
         warning(' Removed first partial trial!');
      else % can we do anything else?
         error(' Cant fix!');
      end
   end   
end
oktr= (bgns<ends);
if ( sum(~oktr)>0 )
   warning('%d end transitions found before start transitions',sum(~oktr));   
   if ( all(~oktr) ) % may be incomplete first/last trial
      if ( all( bgns(1:end-1)<ends(2:end) ) ) % incomplete 1st trial
         bgns=bgns(1:end-1); ends=ends(2:end); epochs=epochs(1:end-1);
         warning(' DISCARDED first partial trial without start time!');
      end
   end
end
if ( any(bgns>ends) ) 
   error('%d start transitions before end transitions',sum(bgns>ends))
end

% deal with other options
if ( ~isempty(opts.fs) ) 
   samp2ms = 1000/opts.fs; 
   ms2samp = opts.fs/1000;
end

% Use the given trial length to over-ride the status info if wanted
if ( ~isempty(opts.trlen_ms) )
   if ( isempty(opts.fs) ) error('no fs: cant compute ms2samp'); end;
   opts.trlen_samp = floor(opts.trlen_ms*ms2samp);
end
if ( ~isempty(opts.trlen_samp) )
   ends=min(bgns+floor(opts.trlen_samp),lastSample);
end

% offset if wanted
if ( ~isempty(opts.offset_ms) ) 
   if ( numel(opts.offset_ms)<2 ) 
      opts.offset_ms=[-opts.offset_ms opts.offset_ms]; 
   end;
   opts.offset_samp = ceil(opts.offset_ms*ms2samp);
end
if ( ~isempty(opts.offset_samp) && ~isequal(opts.offset_ms,0) )
   bgns=bgns+opts.offset_samp(1); 
   ends=min(ends+opts.offset_samp(2),lastSample);
   offset_samp=opts.offset_samp; % zero offset relative to bgn
   badTr= bgns<1 | ends<1 | bgns>lastSample | ends>lastSample;
   if ( any(badTr) )
      warning(sprintf('%d trials of %d #(%s) run outside data-file, deleted',sum(badTr),numel(bgns),sprintf('%d,',find(badTr))));
      bgns=bgns(~badTr);  ends=ends(~badTr);
      bgnsi=bgnsi(~badTr);endsi=endsi(~badTr);
   end
else 
   offset_samp=0;
end
return;