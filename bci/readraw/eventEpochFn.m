function [events,bgns,ends,offset_samp,opts,bgnsi,endsi]=eventEpochFn(events,varargin)
% search through set of events to identify set of epoch start/end points to slice
%
% [events,bgns,ends,offset_samp,opts,bgnsi,endsi]=eventEpochFn(events,varargin)
%
% Inputs:
%  events - [struct] vector of events to search.  Events should have the 
%           fields: 
%               'type','value','sample','offset','duration'
%            OR
%               'TYP',  'VAL', 'POS',  'DUR'
%
% Options:
%  startSet -- {2x1} cell array of cell arrays to match
%              startSet{1} -- type match possibilities
%              startSet{2} -- value match possibilities
%              e.g. to match type='stim' and value=10 or 11 use: 'startSet',{{'stim'} [10 11]}
%        OR
%              [Nx1 float] set of marker values to match.  Any type with this value matches.
%  endSet   -- {2x1} cell array of cell arrays to match as for startSet
%                N.B. If not given then then is the same as start set
%  offset_event   -- {2x1} cell array of to match as for startSet,
%                    the first event matching this after each startSet event is time-zero
%  offset_ms/samp -- [2x1] start/end of window offsets to impose such that
%               win_start = win_start+offset(1), win_end=win_end+offset(2)
%  trlen_ms/samp  -- [1x1] override endSet and just set the epoch endpoint to this much after the start
%  fs         -- [int] sample rate of the data
% Outputs:
%  events-- [M x 1 struct] event which contained the start of epoch info
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
%  bgnsi-- [M x 1] indicies in to the events structure of the start of the epochs
%----------------------------------------------------------------------------
% Identify the trail start/ends
%----------------------------------------------------------------------------
opts=struct('startSet',[],'endSet',[],'offset_event',[],'offset_ms',[],'offset_samp',[],'trlen_ms',[],'trlen_samp',[],'fs',[]);
%if ( numel(varargin)>0 && isstr(varargin{1}) && strcmp('RecPhaseLimits',varargin{1})) varargin(1)=[]; end;
[opts,varargin]=parseOpts(opts,varargin);
if ( numel(varargin)>0 ) opts.startSet=varargin{1}; end;
if ( numel(varargin)>1 ) opts.endSet  =varargin{2}; end;

startSet=opts.startSet; endSet=opts.endSet;
if ( isempty(endSet) ) 
   if ( iscell(startSet) && numel(startSet)>2 ) endSet=startSet(3:end); startSet=startSet(1:2); 
   elseif ( isnumeric(startSet) ) endSet=-startSet;
   else endSet=startSet;
   %else error('Dont know how to generate the endSet');
   end
end
if ( isnumeric(startSet) ) startSet={[] startSet}; end; % only match on the value
if ( ~iscell(startSet) ) startSet={startSet}; end;
if ( ~iscell(endSet) )   endSet  ={endSet}; end;
if ( isempty(endSet) )   endSet  =startSet; end;

if ( isempty(events) ) bgns=[]; ends=[]; return; end

if ( isfield(events,'TYP') )          type=events.TYP;
elseif ( isfield(events,'type') )     type={events.type}; 
else error('couldnt identify event type field');
end
if ( isfield(events,'CodeDesc') ) % convert type to it's correct description
  tmp=type;type={};
  for ei=1:numel(tmp); type{ei} = events.CodeDesc{tmp(ei)}; end
end
if ( isfield(events,'VAL') )          value=events.VAL;
elseif ( isfield(events,'value') )    value={events.value}; 
else                                  value=[];
end;
if ( isfield(events,'POS') )          samp=events.POS;
elseif ( isfield(events,'sample') )   samp=[events.sample]; 
end;
if ( isfield(events,'DUR') )          dur=events.DUR;
elseif ( isfield(events,'duration') ) dur=[events.duration]; 
end;

bgnsi=find(matchEvents(type,value,startSet{:}));
bgns=samp(bgnsi);  % beginning each trial = transition to bgn phase
if ( ~isempty(startSet{1}) ) types=type(bgnsi); else types=value(bgnsi); end;

endsi=[];
if ( isempty(opts.trlen_ms) )
  endsi=find(matchEvents(type,value,endSet{:}));
  ends=samp(endsi)+dur(endsi); % end each trial = transtion *from* end phase
else
  ends=bgns+1;
  endsi=bgnsi;
end

% make sure all the bgns are before the ends!
if ( numel(bgns)~=numel(ends) ) 
   warning('Unequal numbers of start and end transitions: %d~=%d',numel(bgns),numel(ends));
   if ( numel(bgns)>numel(ends) ) 
      % more bgns, may be an incomplete last trial, complete it if so
      if ( bgns(numel(ends)+1)>ends(end) )         
         bgns(numel(ends)+2:end)=[]; types(numel(ends)+2:end)=[];
         ends(end+1)=nsamp;
         warning(' FIXED last partial trial -- MAY BE CORRUPT!');
      else % can we do anything else?
         error(' Cant fix!');
      end
   elseif ( numel(ends) > numel(bgns) ) 
      % more ends, may be an incomplete first trial incomplete, complete if so
      if ( bgns(1)>ends(1) )
         bgns=[1 bgns]; types=[-0 types];
         warning(' FIXED first partial trial -- MAY BE CORRUPT!');
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
         bgns=bgns(1:end-1); ends=ends(2:end); types=types(1:end-1);
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

% offset start point if wanted
if ( ~isempty(opts.offset_event) ) 
  if ( ~iscell(opts.offset_event) ); opts.offset_event={opts.offset_event}; end
  strti=find(matchEvents(type,value,opts.offset_event{:}));  
  for ei=1:numel(bgns);
		mi = find(samp(strti)>=bgns(ei),1); % find first matching event after the bgn event
		if ( ~isempty(mi) )  % shift start position info to this event
		  bgns(ei) = samp(strti(mi));
		end
  end
end


% Use the given trial length to over-ride the status info if wanted
if ( ~isempty(opts.trlen_ms) )
   if ( isempty(opts.fs) ) error('no fs: cant compute ms2samp'); end;
   opts.trlen_samp = floor(opts.trlen_ms*ms2samp);
end
if ( ~isempty(opts.trlen_samp) )
   ends=bgns+floor(opts.trlen_samp);
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
else 
   offset_samp=0;
end

% check is all within the file bounds
badTr= bgns<0 | ends<1 | bgns>max(samp) | ends>max(samp);
if ( any(badTr) )
  warning(sprintf('%d trials of %d #(%s) run outside data-file',sum(badTr),numel(bgns),sprintf('%d,',find(badTr))));
  %bgns=bgns(~badTr);  ends=ends(~badTr);
  %bgnsi=bgnsi(~badTr);endsi=endsi(~badTr);
end

if ( numel(events)>1 )
   events=events(bgnsi);
else % build events structure
   tmp=events;
   events=repmat(struct('type',[],'value',[],'sample',-1,'offset',0,'duration',0),1,numel(bgnsi));
   for bi=1:numel(bgnsi);
      ei=bgnsi(bi);
      if ( iscell(type) ) events(bi).type=type{ei}; else events(bi).type=type(ei); end;
      if ( ~isempty(value) ) if ( iscell(value) ) events(bi).value=value{ei}; else events(bi).value=value(ei); end; end;
      events(bi).sample=samp(ei);
   end
end

return;


function mi=matchEvents(type,value,mtype,mval)
if ( nargin<3 )                   mtype='*'; end;
if ( nargin<4 || isempty(mval) )  mval='*'; end;
if ( (isempty(type) && isempty(value)) || isempty(mtype) || isempty(mval) ) mi=[]; return; end; % fast path!
if ( isstr(mtype) && ~isequal(mtype,'*') ) mtype={mtype}; end;
if ( isstr(mval) && ~isequal(mval,'*') )   mval={mval}; end;
if ( iscell(mtype) && numel(mtype)==2 && isempty(mval) ) 
  mvals=mtype{2}; mtypes=mtype{1};
end;
% find matching types
mi=true(size(type));
if ( isequal(mtype,'*') )
elseif ( isnumeric(mtype) )
  mi=any(repop(type(mi),'==',mtype(:)'),2);
elseif ( iscell(mtype) && isstr(mtype{1}) )
  mi(:)=false;
  for ei=1:numel(mi);
    estr=type(ei);
    for vi=1:numel(mtype);
      mstr=mtype{vi};
      if ( strcmp(estr,mstr) || ... % normal match || prefix match
           ( mstr(end)=='*' && numel(estr)>=numel(mstr)-1 && strcmp(estr(1:numel(mstr)-1),mstr(1:end-1))) )
        mi(ei)=true; break; 
      end
    end
  end
end
% find matching values
if ( isequal(mval,'*') )
elseif ( isnumeric(mval) )
  mi(mi)=any(repop(value(mi),'==',mval(:)'),2);
elseif ( iscell(mval) && isstr(mval{1}) )
  ms =find(mi);
  mi(ms)=false;
  for ei=ms;
    vstr=value(ei);
    for vi=1:numel(mval);
      mstr=mval{vi};
      if ( strcmp(vstr,mstr) || ... % normal match || prefix match
           ( mstr(end)=='*' && numel(vstr)>=numel(mstr)-1 && strcmp(vstr(1:numel(mstr)-1),mstr(1:end-1))) ) 
        mi(ei)=true; break; 
      end;     
    end
  end
end
return;
