function [type,bgns,ends,offset_samp,opts]=comprecphases(PresentationPhase,varargin)
% Identify the trail start/ends
%
% [bgns,ends]=comprecphases(PresentationPhase)
%
% Inputs:
%  PresentationPhase -- [Nx1] vector of presentation phase values at each sample
% Options:
%  startSet    -- [Ms x 1]  any-value in a start pair can indicate rec-start
%                       and any matching value in end the same
%                       +ve -> match at the rising edge, i.e. when this phase is entered
%                       -ve -> match at the falling edge, i.e. when this phase is finished
%               OR {[Msx1]} cell array of start Sets
%  endSet      -- [Me x 1] similar to start set but indicates the end of a recording phase
%                   (-startSet)
%               OR {[Msx1]} cell array of end Sets
%  offset_ms/samp -- [2x1] start/end of window offsets to impose such that
%               win_start = win_start+offset(1), win_end=win_end+offset(2)
%  trlen_ms/samp  -- [1x1] override endSet and just set the epoch endpoint to this much after the start
%  fs         -- [int] sample rate of the data
% Outputs:
%  type -- [M x 1] id of the epoch (start marker ID)
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
opts=struct('startSet',[],'endSet',[],'offset_ms',[],'offset_samp',[],'trlen_ms',[],'trlen_samp',[],'fs',[]);
if ( numel(varargin)>0 && isstr(varargin{1}) && strcmp('RecPhaseLimits',varargin{1})) varargin(1)=[]; end;
[opts,varargin]=parseOpts(opts,varargin);
if ( numel(varargin)>0 ) opts.startSet=varargin{1}; end;
if ( numel(varargin)>1 ) opts.endSet  =varargin{2}; end;
if ( numel(varargin)>2 ) warning(sprintf('%d Unrecognised options',numel(varargin)-2)); end;

startSet=opts.startSet; endSet=opts.endSet;
if ( isempty(endSet) ) 
   if ( iscell(startSet) ) endSet=startSet(2:2:end); startSet=startSet(1:2:end); 
   elseif ( isnumeric(startSet) ) endSet=-startSet;
   else error('Dont know how to generate the endSet');
   end
end
if ( ~iscell(startSet) ) startSet={startSet}; end;
if ( ~iscell(endSet) )   endSet  ={endSet}; end;
if ( isempty(endSet) )   endSet  =startSet; end;
if ( ~isempty(opts.trlen_ms) || ~isempty(opts.trlen_samp) ) endSet={[]}; end;

% Now we can extract the bits we want
bgns=[]; ends=[];
for j=1:numel(startSet);  
   [tbgns tends]=compRecPhasesInner(PresentationPhase,startSet{j},endSet{min(end,j)});
   bgns=[bgns  tbgns];
   ends=[ends  tends];
end
% sort by bgn point
[bgns,si]=sort(bgns,'ascend'); ends=ends(si); 
type = PresentationPhase(bgns); % record epoch type

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
   badTr= bgns<1 | ends<1 | bgns>numel(PresentationPhase) | ends>numel(PresentationPhase);
   if ( any(badTr) )
      warning(sprintf('%d trials of %d #(%s) run outside data-file, deleted',sum(badTr),numel(bgns),sprintf('%d,',find(badTr))));
      bgns=bgns(~badTr); ends=ends(~badTr);
   end
else 
   offset_samp=0;
end
return;

function [bgns,ends]=compRecPhasesInner(PresentationPhase,startSet,endSet)
if ( nargin < 3 ) endSet=[]; end;

rec=false(size(PresentationPhase));
for i=1:numel(startSet);% where PresentationPhase(1) matchs vals
   phaseVal=startSet(i);
   tmp= PresentationPhase == abs(phaseVal);
   if( phaseVal>0 )     rec([false;diff(tmp(:))>0])=true;  % element *after* rising edge
   elseif( phaseVal<0 ) rec([false;diff(tmp(:))<0])=true;  % element *after* falling edge
   else warning('0 as a marker isnt supported');
   end
end
bgns=find(rec); bgns=bgns(:);     % beginning each trial = 1st sample in new phase

if ( ~isempty(endSet) )
  rec=false(size(PresentationPhase));
  for i=1:numel(endSet);% where PresentationPhase(2) matchs vals
    phaseVal=endSet(i);
    tmp= PresentationPhase == abs(phaseVal);
    if( phaseVal>0 )     rec([false;diff(tmp(:))>0])=true;  % element *after* rising edge
    elseif( phaseVal<0 ) rec([false;diff(tmp(:))<0])=true;  % element *after* falling edge
    else warning('0 as a marker isnt supported');
    end
  end
  ends=find(rec);  ends=ends(:)-1;   % end each trial = last sample in old phase
else
  ends=bgns+1;
end

% make sure all the bgns are before the ends!
if ( isempty(bgns) || isempty(ends) ) 
   bgns=[]; ends=[];
elseif ( numel(bgns)~=numel(ends) ) 
   warning('Unequal numbers of start and end transitions: %d~=%d',numel(bgns),numel(ends));
   % use the closest bgn before each end AND exclude cases where is end in between
   bgnrec=false(size(bgns)); endrec=false(size(ends));
   for i=1:numel(ends); 
      bgni=find(bgns<ends(i),1,'last'); endi=find(ends<ends(i),1,'last');
      if(~isempty(bgni) && (isempty(endi) || ends(endi)<bgns(bgni)) ) 
         bgnrec(bgni)=true; endrec(i)=true;
      end;
   end
   if ( ~any(bgnrec) || ~any(endrec) ) error('Cant fix!');
   else
      if( any(~bgnrec) ) warning('FIXED extra bgns'); end;
      if( any(~endrec) ) warning('FIXED extra ends'); end;
      bgns=bgns(bgnrec); ends=ends(endrec);
   end
end
oktr= (bgns<=ends);
if ( sum(~oktr)>0 )
   if ( all(~oktr) ) % may be incomplete first/last trial
      if ( all( bgns(1:end-1)<ends(2:end) ) ) % incomplete last trial
         ends=[ends(2:end); numel(PresentationPhase)];
         warning(' FIXED partial last trial -- MAY BE CORRUPT');
      end
   else
      warning('%d Start transitions found before end transitions',sum(~oktr));
   end
end
if ( any(bgns>ends) ) 
   warning('%d start transitions before end transitions',sum(bgns>ends))
end
return;

%-------------------------------------------------------------------------------------
function testCases()

N=1000; M=10;
Y=zeros(N,1); Y(round(rand(M,1)*N))=round(rand(M,1)*M); % make rand marker channel
tt=[1;find(Y~=0)]; for i=2:numel(tt); Y(tt(i-1)+1:tt(i))=Y(tt(i)); end;

[bg,en]=compRecPhases(Y,5,5)
[bg,en]=compRecPhases(Y,5,-5)
