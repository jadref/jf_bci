function [type,bgns,ends,offset_samp,opts]=selectEvents(events,varargin)
% Identify the trail start/ends
%
% [bgns,ends]=selectEvents(events,...)
%
% Inputs:
%  events -- [struct Nx1] vector of event structures with fields:
%               'TYP',  'VAL', 'POS',  'DUR'
% Options:
%  eventTypes -- {str} set of event types to keep
%  fs         -- [int] sample rate of the data
%  offset_ms  -- [2x1 int] offset time in ms before/after to include in trial
%  trlen_ms   -- [int] Force trials to be this long (from start marker)
% Outputs:
%  type -- [M x 1] id of the epoch (start marker ID)
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
opts=struct('eventTypes',[],'offset_ms',[],'offset_samp',[],'trlen_ms',[],'trlen_samp',[],'fs',[]);
opts=parseOpts(opts,varargin);

eventTypes=opts.eventTypes;
if ( iscell(eventTypes) && ischar(eventTypes{1}) && isnumeric(events.TYP) ) % convert strings to eventIDs
  if ( isfield(events,'CodeDesc') ) typeDesc=events.CodeDesc;
  elseif( isfield(events,'typeDesc') ) typeDesc=events.typeDesc;
  else error('couldnt find a type description in the event structure');
  end
  for i=1:numel(eventTypes);
    tt=strmatch(eventTypes{i},typeDesc);
    if ( ~isempty(tt) ) eventTypes{i}=tt(1);
    else error('didnt match event type: %s',eventTypes{i});
    end
  end
end

% Now we can extract the bits we want
bgns=[]; ends=[]; type=[];
for i=1:numel(eventTypes);
  if ( iscell(eventTypes) ) ev=eventTypes{i}; else ev=eventTypes(i); end;
  if ( isnumeric(ev) )
    eventIdx=find(events.TYP==ev);
  else
    eventIdx=[];
    for k=1:numel(events.TYP);
      if ( iscell(events.TYP) ) evk=events.TYP{k}; else evk=events.TYP(k); end;
      if ( isequal(evk,ev) )    eventIdx=[eventIdx k];  end
    end
  end
  tbgns   =[events.POS(eventIdx)];
  tends   =tbgns+[events.DUR(eventIdx)];
  if ( isnumeric(ev) ) ttype   = repmat(ev,size(tbgns)); 
  else                 ttype   = repmat(j,size(tbgns)); % BODGE!!!
  end;
  bgns=[bgns; tbgns(:)];
  ends=[ends; tends(:)];
  type=[type; ttype(:)];
end
% sort by bgn point
[bgns,si]=sort(bgns,'ascend'); ends=ends(si); type=type(si);

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
   ends=min(bgns+floor(opts.trlen_samp),numel(PresentationPhase));
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
   ends=ends+opts.offset_samp(2);   
   offset_samp=opts.offset_samp; % zero offset relative to bgn
   badTr= ends<1 | bgns>max(events.POS);
   if ( any(badTr) )
      warning(sprintf('%d trials of %d #(%s) run outside data-file, deleted',sum(badTr),numel(bgns),sprintf('%d,',find(badTr))));
      bgns=bgns(~badTr); ends=ends(~badTr);
   end
else 
   offset_samp=0;
end
return;

%-------------------------------------------------------------------------------------
function testCases()

N=1000; M=10;
Y=zeros(N,1); Y(round(rand(M,1)*N))=round(rand(M,1)*M); % make rand marker channel
tt=[1;find(Y~=0)]; for i=2:numel(tt); Y(tt(i-1)+1:tt(i))=Y(tt(i)); end;

[bg,en]=compRecPhases(Y,5,5)
[bg,en]=compRecPhases(Y,5,-5)
